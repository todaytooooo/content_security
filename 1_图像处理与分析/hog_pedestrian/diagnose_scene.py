"""Inspect annotated people through each stage of the existing detector."""
from pathlib import Path
import json
import sys

import cv2
import joblib
import numpy as np

import detect
from train import read_boxes, iou, full_person_crop, feature, hog_descriptor


def main():
    root = Path(__file__).resolve().parent
    source = root / 'data/INRIAPerson_complete/INRIAPerson/Test'
    image = detect.read_image(source / 'pos/crop001661.png')
    truths = read_boxes(source / 'annotations/crop001661.txt')
    model = joblib.load(root / 'models/hog_svm.joblib')
    boxes, scores = detect.candidate_detections(image, model, 4, 0.9)
    selected = [i for i, score in enumerate(scores) if score >= 1.0]
    chosen_boxes = [boxes[i] for i in selected]
    chosen_scores = [scores[i] for i in selected]
    stages = {}

    # Observe the real NMS implementation without changing its behavior.
    def trace(frame, event, arg):
        if frame.f_code is detect.nms.__code__ and event == 'return':
            stages['iou_nms'] = list(frame.f_locals['keep'])
            stages['final'] = list(arg)
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        detect.nms(chosen_boxes, chosen_scores, model.get('nms_overlap', 0.35))
    finally:
        sys.settrace(previous_trace)

    output = root / 'outputs/diagnosis_crop001661'
    output.mkdir(parents=True, exist_ok=True)
    report = {'image_size': [image.shape[1], image.shape[0]],
              'threshold': 1.0, 'stride': 4, 'scale_step': 0.9,
              'candidate_count': len(boxes), 'threshold_count': len(selected),
              'iou_nms_count': len(stages['iou_nms']),
              'final_count': len(stages['final']), 'people': []}
    canvas = image.copy()
    for index, truth in enumerate(truths, 1):
        overlaps = np.asarray([iou(box, truth) for box in boxes])
        best = int(overlaps.argmax())
        matching = np.flatnonzero(overlaps >= 0.5)
        highest = int(matching[np.argmax(np.asarray(scores)[matching])]) if len(matching) else None
        stages_for_person = {}
        for name, indices in [('threshold', range(len(selected))), *stages.items()]:
            matches = [j for j in indices if iou(chosen_boxes[j], truth) >= 0.5]
            stages_for_person[name] = [
                {'box': chosen_boxes[j], 'score': chosen_scores[j]} for j in matches]
        crop_score = float(model['classifier'].decision_function(
            feature(hog_descriptor(), full_person_crop(image, truth))[None, :])[0])
        report['people'].append({
            'id': index, 'truth_xyxy': truth,
            'best_geometry': {'box': boxes[best], 'iou': float(overlaps[best]), 'score': scores[best]},
            'highest_score_iou_ge_05': None if highest is None else {
                'box': boxes[highest], 'iou': float(overlaps[highest]), 'score': scores[highest]},
            'training_style_crop_score': crop_score,
            'stages': stages_for_person})
        x1, y1, x2, y2 = truth
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 150, 0), 2)
        cv2.putText(canvas, f'GT {index}', (x1, y1 - 8), 0, 0.7, (255, 150, 0), 2)
        x, y, w, h = boxes[highest if highest is not None else best]
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 180, 255), 2)
    for j in stages['final']:
        x, y, w, h = chosen_boxes[j]
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 220, 0), 2)
        cv2.putText(canvas, f'{chosen_scores[j]:.2f}', (x, max(18, y-5)), 0, 0.6, (0, 220, 0), 2)
    cv2.imencode('.png', canvas)[1].tofile(str(output / 'diagnosis.png'))
    text = json.dumps(report, indent=2)
    (output / 'report.json').write_text(text, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
