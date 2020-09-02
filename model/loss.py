import torch
import torch.nn as nn
from config import DefaultConfig

def coords_fmap2orig(feature, stride):

    h, w = feature.shape[1:3]
    shift_x = torch.arange(0, w*stride, stride, dtype=torch.float32)
    shift_y = torch.arange(0, h*stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])

    coords = torch.stack([shift_x, shift_y], -1) + stride // 2

    return coords

class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        sper().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        """
        inputs: 
        [cls, cnt, reg]
        [gt_boxes]
        [gt_classes]
        
        return:
        cls_targets
        cnt_targets
        reg_targets
        """

        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]

        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []

        assert len(self.strides) == len(cls_logits)

        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level], self.limit_range[level])

            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level,dim=1), torch.cat(cnt_targets_all_level,dim=1), torch.cat(reg_targets_all_level,dim=1)

    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        """
        gt_boxes: [batch, m, 4]
        classes: [batch, m]
        """
        
        cls_logits,cnt_logits,reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]

        m = gt_boxes.shape[1]

        cls_logits = cls_logits.permute(0,2,3,1) #[batch, h, w, class]

        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2]

        cls_logits = cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]  
        cnt_logits = cnt_logits.permute(0,2,3,1)
        cnt_logits = cnt_logits.reshape((batch_size,-1,1))

        reg_preds=reg_preds.permute(0,2,3,1)
        reg_preds=reg_preds.reshape((batch_size,-1,4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = corrds[:, 1]

        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]

        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[batch_size,h*w,m,4]

        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[batch_size,h*w,m]

        off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
        off_max=torch.max(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]

        mask_in_gtboxes=off_min>0
        mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])

        radiu=stride*sample_radiu_ratio

        gt_center_x=(gt_boxes[...,0]+gt_boxes[...,2])/2
        gt_center_y=(gt_boxes[...,1]+gt_boxes[...,3])/2

        c_l_off=x[None,:,None]-gt_center_x[:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off=y[None,:,None]-gt_center_y[:,None,:]
        c_r_off=gt_center_x[:,None,:]-x[None,:,None]
        c_b_off=gt_center_y[:,None,:]-y[None,:,None]
        c_ltrb_off=torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[batch_size,h*w,m,4]
        c_off_max=torch.max(c_ltrb_off,dim=-1)[0]
        mask_center=c_off_max<radiu

        mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]

        areas[~mask_pos]=99999999
        areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]

        reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))#[batch_size,h*w,4]

        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]#[batch_size,h*w,m]
        cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])

        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]

        #process neg location
        mask_pos_2=mask_pos.long().sum(dim=-1)#[batch_size,h*w]
        mask_pos_2=mask_pos_2>=1

        cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]
        cnt_targets[~mask_pos_2]=-1
        reg_targets[~mask_pos_2]=-1

        return cls_targets,cnt_targets,reg_targets


def compute_cls_loss(preds, targets, mask):
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)

    num_pos = torch.sum(mask, dim=[1,2]).clamp_(min=1).float() #[batch_size,]

    for pred in preds:
        pred = pred.permute(0,2,3,1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)

    preds = torch.cat(preds_reshape, dim=1)

    loss = []

    for batch_i in range(batch_size):
        pred_pos = preds[batch_i]
        target_pos = targets[batch_i]
        target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
        loss.append(focal_loss_from_logits(pred_pos, target_pose).view(1))

    return torch.cat(loss, dim=0) / num_pos #[batch,]



