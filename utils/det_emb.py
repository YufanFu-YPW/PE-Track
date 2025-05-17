import pickle
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from external.YOLOX.yolox.models.build import yolox_custom
from utils.embedding import EmbeddingComputer

class DetEmb:
    def __init__(self, use_emb=True):
        self.use_emb = use_emb
        self.det_model = None
        self.emb_model = None
        self.cache_data = {}

    def load_models(self, det_ckpt, det_exp_path, emb_dataset, emb_test, device='cuda:1'):
        """
        Load the model
        det_ckpt:
        det_exp_path:
        emb_dataset:
        emb_test: (True) represents the test set; (False) Represents the validation set
        """
        det_model = yolox_custom(ckpt_path=det_ckpt, exp_path=det_exp_path, device='cuda:0').eval()
        self.det_model = det_model.half()
        if self.use_emb:
            self.emb_model = EmbeddingComputer(emb_dataset, emb_test, grid_off=True, device=device)

    def load_cache(self, cache_path):
        with open(cache_path, "rb") as fp:
            self.cache_data = pickle.load(fp)

    def get_data_from_models(self, img, det_img_size, det_conf_thrd, det_nms_thrd, tag):
        ori_h, ori_w = img.shape[:2]
        input_img = self.process_image(img, det_img_size)[0]
        input_img = torch.as_tensor(input_img).type(torch.cuda.HalfTensor).unsqueeze(0)
        outputs = self.det_model(input_img)
        outputs = self.postprocess(outputs, det_conf_thrd, det_nms_thrd)[0]  # [x1,y1,x2,y2]
        if outputs is not None:
            outputs = self.postprocess_results(outputs, det_img_size, ori_h, ori_w)  # [x1,y1,x2,y2]
            dets = outputs[:, :7].detach().cpu().numpy()
            dets = self.area_filter(dets, ori_h, ori_w)  # [x1,y1,x2,y2]
        else:
            dets = np.array([])

        dets_embs = None
        if self.use_emb:
            dets_embs = np.ones((dets.shape[0], 1))
            if dets.shape[0] != 0:
                det_boxs = dets[:, :4].copy()
                dets_embs = self.emb_model.compute_embedding(img, det_boxs, tag)
            
        self.cache_data[tag] = {'dets': dets,
                                'det_embs': dets_embs}
        return dets, dets_embs

    def get_data_from_cache(self, tag):
        data = self.cache_data[tag]
        return data['dets'], data['det_embs']  # dets, embs

    def dump_cache(self, save_cache_path):
        with open(save_cache_path, "wb") as fp:
            pickle.dump(self.cache_data, fp)


    def fuse_conv_and_bn(self, conv, bn):
        # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
        fusedconv = (
            nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = (
            torch.zeros(conv.weight.size(0), device=conv.weight.device)
            if conv.bias is None
            else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

    def fuse_model(self, model):
        from yolox.models.network_blocks import BaseConv

        for m in model.modules():
            if type(m) is BaseConv and hasattr(m, "bn"):
                m.conv = self.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        return model


    def process_image(self, image, input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self, prediction, conf_thre, nms_thre, num_classes=1):
        """
        Filter the detection results, non-maximum suppression
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def postprocess_results(self, outputs, img_size, ori_h, ori_w):
        bboxes = outputs[:, 0:4]

        # preprocessing: resize
        scale = min(
            img_size[0] / float(ori_h), img_size[1] / float(ori_w)
        )
        bboxes /= scale

        clses = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]

        obj_conf = outputs[:, 4]
        cls_conf = outputs[:, 5]

        return torch.concat((bboxes, scores[:, None], obj_conf[:, None], cls_conf[:, None], clses[:, None]), dim=1)

    def area_filter(self, dets, ori_h, ori_w, area=0):
        res = []
        bbox = dets[:, 0:4].copy()
        bbox = np.round(bbox).astype(np.int32)
        bbox[:, 0] = bbox[:, 0].clip(0, ori_w)
        bbox[:, 1] = bbox[:, 1].clip(0, ori_h)
        bbox[:, 2] = bbox[:, 2].clip(0, ori_w)
        bbox[:, 3] = bbox[:, 3].clip(0, ori_h)

        for i in range(0, bbox.shape[0]):
            b = bbox[i]
            if (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0 and (b[2] - b[0]) * (b[3] - b[1]) > area:
                res.append(dets[i])
        
        res = np.array(res)

        return res





