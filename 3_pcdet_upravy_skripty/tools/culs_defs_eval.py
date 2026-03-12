import argparse, os, logging, pickle, time, torch, scipy, pcdet, math
import numpy as np
import sklearn.metrics
from easydict import EasyDict
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def get_gt_preds_loss(cfg, model, dataloader):
    model.eval()
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.0
    ground_truths, predictions, loss = get_data_and_loss(model, dataloader)
    return ground_truths, predictions, loss
    

def evaluate_model_loss(cfg, model, dataloader, pr_curve=False):
    model.eval()
    _, predictions, loss = get_data_and_loss(model, dataloader)
    return predictions, {'loss': loss}


def evaluate_model_scores(cfg, model, dataloader):
    tick_1 = time.perf_counter()
    metrics = {}
    model.eval()
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.0
    if cfg.MODEL.NAME in ["SECONDNet", "PillarNet"]:
        ground_truths, predictions, losses = get_data_and_loss(model, dataloader)
    elif cfg.MODEL.NAME == "PointRCNN":
        ground_truths, predictions, losses = get_point_data_and_loss(model, dataloader)
    else:
        raise Exception("Can get loss results only for SECONDNet and PointRCNN models. If that's okay, use the get_data_only() function.")
    metrics.update(losses)
    tick_2 = time.perf_counter()
    
    scores = [x / 100.0 for x in range(1, 100, 1)]
    
    metrics.update(
        get_count_metrics(
            ground_truths, predictions, scores
        )
    )
    
    tick_3 = time.perf_counter()
    metrics.update(
        get_detection_metrics(
            ground_truths, predictions, scores, cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
        )
    )
    tick_4 = time.perf_counter()
    metrics['eval_time'] = tick_4 - tick_1
    # print(f"P/C/D/T: {tick_2 - tick_1:.2f} + {tick_3 - tick_2:.2f} + {tick_4 - tick_3:.2f} = {tick_4 - tick_1:.2f}")
    return predictions, metrics


def get_detection_metrics(ground_truths, predictions, confidence_thresholds, iou_thresholds):
    result = {
        "detection_per_cf": {}
    }
    confidence_scores = [np.array(x['score']) for x in predictions]
    iou = get_raw_iou(ground_truths, predictions, confidence_thresholds[0])
    iou = [np.array(x) for x in iou]
    first = True
    for cf_t in confidence_thresholds:
        drop_low_confidences_detection(iou, confidence_scores, cf_t)
        score_result = {}
        for iou_t in iou_thresholds:
            sum_tp, sum_fp, sum_fn = 0, 0, 0
            samples_p, samples_r, samples_f = [], [], []
            for sample in iou:
                sample_tp, sample_fp, sample_fn = iou_to_truths(sample, iou_t)
                sample_p, sample_r, sample_f = get_pr_and_score(sample_tp, sample_fp, sample_fn)
                sum_tp += sample_tp
                sum_fp += sample_fp
                sum_fn += sample_fn
                samples_p.append(sample_p)
                samples_r.append(sample_r)
                samples_f.append(sample_f)
            sum_p, sum_r, sum_f = get_pr_and_score(sum_tp, sum_fp, sum_fn)
            score_result[f"precis_sum_{iou_t}"] = sum_p
            score_result[f"recall_sum_{iou_t}"] = sum_r
            score_result[f"fscore_sum_{iou_t}"] = sum_f
            avg_p, avg_r, avg_f = np.average(samples_p), np.average(samples_r), np.average(samples_f)
            score_result[f"precis_avg_{iou_t}"] = avg_p
            score_result[f"recall_avg_{iou_t}"] = avg_r
            score_result[f"fscore_avg_{iou_t}"] = avg_f
            if first:
                result[f"fscore_sum_{iou_t}"] = sum_f
                result[f"fscore_avg_{iou_t}"] = avg_f
                result[f"fscore_sum_{iou_t}_score"] = cf_t
                result[f"fscore_avg_{iou_t}_score"] = cf_t
            else:
                for key, var in zip(["fscore_sum", "fscore_avg"], [sum_f, avg_f]):
                    if var > result[f"{key}_{iou_t}"]:
                        result[f"{key}_{iou_t}"] = var
                        result[f"{key}_{iou_t}_score"] = cf_t
        first = False
        result['detection_per_cf'].update({cf_t: score_result})
    for iou_t in iou_thresholds:
        for typ in ['sum', 'avg']:
            cf_t = result[f"fscore_{typ}_{iou_t}_score"]
            for pr in ['precis', 'recall']:
                    result[f"{pr}_{typ}_{iou_t}"] = result["detection_per_cf"][cf_t][f"{pr}_{typ}_{iou_t}"]
            ap = 0
            cf_len = len(confidence_thresholds)
            det_dict = result['detection_per_cf']
            for i in range(cf_len):
                ri = det_dict[confidence_thresholds[i]][f"recall_{typ}_{iou_t}"]
                if i == cf_len - 1:
                    rj = 0.0
                else:
                    rj = det_dict[confidence_thresholds[i+1]][f"recall_{typ}_{iou_t}"]
                pi = det_dict[confidence_thresholds[i]][f"precis_{typ}_{iou_t}"]
                ap += (ri - rj) * pi
            result[f'ap_{typ}_{iou_t}'] = ap
            """
            ap_bad = 0
            for i in range(len(confidence_thresholds)-1):
                cf_curr = confidence_thresholds[i]
                cf_next = confidence_thresholds[i+1]
                sub_curr = result['detection_per_cf'][cf_curr]
                sub_next = result['detection_per_cf'][cf_next]
                ap_bad += (sub_curr[f"recall_{typ}_{iou_t}"] - sub_next[f"recall_{typ}_{iou_t}"]) * sub_curr[f"precis_{typ}_{iou_t}"]
            result[f'apBAD_{typ}_{iou_t}'] = ap_bad
            """
    return result



def get_raw_iou(ground_truths, predictions, min_score):
    result = []
    for sample_gt, sample_pred in zip(ground_truths, predictions):
        boxes_gt = [np.asarray(x[:-1].to('cpu')) for x in sample_gt['gt_boxes'][0]]
        boxes_pred = sample_pred['boxes_lidar']
        boxes_gt = torch.tensor(boxes_gt, device='cuda', dtype=torch.float32)
        boxes_pred = torch.tensor(boxes_pred, device='cuda', dtype=torch.float32)
        iou = boxes_iou3d_gpu(boxes_gt, boxes_pred).to('cpu')
        result.append(iou)
    return result


def drop_low_confidences_detection(iou, scores, threshold):
    for i in range(len(scores)):
        newlen = len([x for x in scores[i] if x >= threshold])
        scores[i] = scores[i][:newlen]
        iou[i] = iou[i][:,:newlen]
        
        

def get_count_metrics(ground_truths, predictions, confidence_thresholds):
    result = { 'count_per_cf': {} }
    num_gt = np.array([len(x['gt_boxes'][0]) for x in ground_truths])
    confidence_scores = [np.array(x['score']) for x in predictions]
    b_rsq, b_rmse, b_mae = 0, math.inf, math.inf
    s_rsq, s_rmse, s_mae = 0, 0, 0
    for cf_t in confidence_thresholds:
        drop_low_confidences_count(confidence_scores, cf_t)
        num_ps = np.array([len(x) for x in confidence_scores])
        rsq = scipy_correlation(num_gt, num_ps) ** 2
        rmse = sklearn.metrics.root_mean_squared_error(num_gt, num_ps)
        mae = sklearn.metrics.mean_absolute_error(num_gt, num_ps)
        score_result = {
            'rmse': rmse,
            'mae': mae,
            'rsq': rsq,
        }
        result['count_per_cf'].update({f"{cf_t}": score_result})
        if rmse < b_rmse:
            b_rmse = rmse
            s_rmse = cf_t
        if mae < b_mae:
            b_mae = mae
            s_mae = cf_t
        if rsq > b_rsq:
            b_rsq = rsq
            s_rsq = cf_t
    result['rmse'] = b_rmse
    result['mae'] = b_mae
    result['rsq'] = b_rsq
    result['rmse_score'] = s_rmse
    result['mae_score'] = s_mae
    result['rsq_score'] = s_rsq
    return result


def drop_low_confidences_count(scores, threshold):
    for i in range(len(scores)):
        newlen = len([x for x in scores[i] if x >= threshold])
        scores[i] = scores[i][:newlen]

def scipy_correlation(A, B):
    try:
        result = scipy.stats.pearsonr(A, B).statistic
    except ValueError:
        return 0.0
    if math.isnan(result):
        return 0.0
    return result

# Much faster than SciPy implementation
# https://stackoverflow.com/a/71847068
def pairwise_correlation(A, B):
    if A.shape != B.shape: #
        return 0.0
    if np.any(A == B): # constant array input
        return 1.0 # not likely - all prediction counts match ground truth counts
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return (
        am.T @ bm / (
            np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T *
            np.sqrt(np.sum(bm**2, axis=0, keepdims=True))
        )
    )[0]


def get_data_and_loss(model, dataloader):
    ground_truths, predictions = [], []
    losses = np.empty(len(dataloader), dtype=np.float64)
    for i, sample_gt in enumerate(dataloader):
        sample_pred = make_prediction(model, sample_gt, dataloader.dataset)
        ground_truths.append(sample_gt)
        predictions.append(sample_pred[0])
        losses[i] = model.dense_head.get_loss()[1]['rpn_loss']
    return ground_truths, predictions, {"loss": np.average(losses)}

def get_point_data_and_loss(model, dataloader):
    ground_truths, predictions = [], []
    losses = np.empty(len(dataloader), dtype=np.float64)
    losses_point = np.empty(len(dataloader), dtype=np.float64)
    losses_roi = np.empty(len(dataloader), dtype=np.float64)
    for i, sample_gt in enumerate(dataloader):
        sample_pred = make_prediction(model, sample_gt, dataloader.dataset)
        ground_truths.append(sample_gt)
        predictions.append(sample_pred[0])
        loss_point = model.point_head.get_loss()[0]
        loss_roi = model.roi_head.get_loss()[0]
        loss = loss_point + loss_roi
        losses[i] = loss
        losses_point[i] = loss_point
        losses_roi[i] = loss_roi
    loss_dict = {
        "loss": np.average(losses),
        "loss_point": np.average(losses_point),
        "loss_roi": np.average(losses_roi)
    }
    return ground_truths, predictions, loss_dict

def make_prediction(model, sample_gt, dataset):
    pcdet.models.load_data_to_gpu(sample_gt)
    with torch.no_grad():
        pred_dicts, ret_dict = model(sample_gt)
    return dataset.generate_prediction_dicts(
        sample_gt, pred_dicts, dataset.class_names, None
    )

# slow but more precise - only allows one prediction per ground truth
# when calculating TP.
# from munkres import Munkres
# def iou_to_truths_munkres(iou, threshold):
#     m = Munkres()
#     if iou.shape[0] > iou.shape[1]:
#         transpose = True
#         indexes = m.compute(1 - iou.transpose())
#     else:
#         transpose = False
#         indexes = m.compute(1 - iou)
#     tp = 0
#     if transpose:
#         for row, column in indexes:
#             if iou[column][row] >= threshold:
#                 tp += 1
#     else:
#         for row, column in indexes:
#             if iou[row][column] >= threshold:
#                 tp += 1
#     return tp, iou.shape[1]-tp, iou.shape[0]-tp

# Converts the IoU matrix returned by PCDet's boxes_iou3d_gpu into the
# absolute number of true positives, false positives, and false negatives.
# BUGGY AND WRONG!
# def iou_to_truths(iou, threshold):
#     tp = 0
#     # fp = 0
#     fn = 0
#     for gt in iou:
#         if max(gt) > threshold:
#             tp += 1
#         else:
#             fn += 1
#     # Faulty calculation here!
#     # When two predictions detect an object, one of them is a FP!
#     # This calculation wouldn't increase FP in that case.
#     # Also it's slower.
#     # for pred in iou.transpose(0, 1):
#     #     if max(pred) <= threshold:
#     #         fp += 1
#     fp = iou.shape[1]-tp
#     return tp, fp, fn


def iou_to_truths(iou, threshold):
    init_gts, init_preds = iou.shape
    if init_preds == 0:
        return 0, 0, init_gts
    tp = 0
    fn = 0
    for igt in range(init_gts):
        if iou.shape[1] == 0:
            fn += init_gts - igt
            break
        gt = iou[igt]
        imax = np.argmax(gt)
        if gt[imax] > threshold:
            tp += 1
            iou = np.delete(iou, imax, 1)
        else:
            fn += 1
    fp = iou.shape[1]
    return tp, fp, fn


# Calculates precision, recall, and f-score for given numbers of
# true positives, false positives, and false negatives.
def get_pr_and_score(tp, fp, fn):
    if (tp+fp) == 0:
        return 1, 0, 0
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        # Using this DOUBLES the eval time, what the hecc?
        # f = scipy.stats.hmean([p, r])
        f = 2*tp / (2*tp + fp + fn) 
        return p, r, f
