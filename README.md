# Object Detection for Hospital Indoor Dataset (Thesis)
In this repository I provide the code that was used for my experiments in object detection task. This work was done for my thesis accomplishment in Harokopio University (Athens - Greece).

Some SOTA foundation models were used including Grounding Dino, YOLO and YOLOE. Text and visual prompting techniques were utilized in a zero-shot and few-shot manner. Some results are also exposed for comparison.
The dataset is not pubicly available and can be obtained from this repository: https://github.com/Wangmmstar/Hospital_Scene_Data

Below you can see the a bar chart of the results of all experiments
<img width="907" height="506" alt="image" src="https://github.com/user-attachments/assets/f44971b6-c09c-4cc8-b30a-0efb5014a10d" />


| Algorithm	| Prompting *	| mAP	| mAP50	| mAP75 |
| --------- | ----------- | --- | ----- | ----- |
| YOLOv11-L  (trained)	| T	| 0,490	| 0,705	| 0,524 |
| YOLOE 11-L (k=15, dmin=0.6, dmax=1.3)	| V	| 0,297	| 0,445	| 0,320
| YOLOE 11-L (15 random images)	| V	| 0,296	| 0,442	| 0,321
| YOLOE 11-L (k=10, dmin=0.6, dmax=1.3)	| V	| 0,286	| 0,429	| 0,308
| YOLOE 11-L (10 random images)	| V	| 0,284	| 0,425	| 0,306
| YOLOE 11-L (k=15, dmin=0.8, dmax=1.4)	| V	| 0,270	| 0,405	| 0,291
| G. Dino Tiny (zero-shot)	| T	| 0,269	| 0,374	| 0,288
| G. Dino Base (zero-shot)	| T	| 0,264	| 0,359	| 0,287
| YOLOE 11-L (5 random images)	| V | 0,262	| 0,387	| 0,285
| YOLOE 11-L (k=10, dmin=0.8, dmax=1.4)	| V	| 0,257	| 0,385	| 0,277
| YOLOE 11-L (k=5, dmin=0.6, dmax=1.3)	| V	| 0,253	| 0,377	| 0,273
| YOLOE 11-L (k=5, dmin=0.8, dmax=1.4)	| V	| 0,249	| 0,368	| 0,271
| YOLOE 11-L (zero shot, size=1152)	| T	| 0,151	| 0,220	| 0,163
| YOLOE 11-L (zero-shot, size=640)	| T	| 0,140	| 0,197	| 0,152
| YOLOE 11-L (1 random image)	| V	| 0,130	| 0,180	| 0,143
| * T = Text prompting, V = Visual prompting |

