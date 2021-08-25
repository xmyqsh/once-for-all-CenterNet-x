# once-for-all-CenterNet-x
1. Implement CenterNet*/CenterNet2/CenterPoint(PointPillar version) device specialized searched by the once-for-all
2. Design CenterNetF/Probabilistic CenterNetF/CenterPointF(PointPillar version)/Probabilistic CenterPointF(try use
3. LidarRCNN to optimize the CenterNet2's second stage) which is a faster architecture inspired by YOLOF.
4. Use 3 phase training designed by DetNAS and reuse the pretrained model from once-for-all as the first phase.
NOTE: PointPillar Version CenterPoint is fast and accurate enough for small-scale point cloud from edge device which
      does not concern about small/sparse object far away.

# Future Work
1. Be a service in k8s, continuous accumulating data, support online active learning and model training and updating online, accelerate model develop and deployment.
2. Apply once-for-all on VO and VIO, use detection to remove moving forward ground, use loop-closure to calibrate more accurate location, once training on cloud, destributing to diverse edge devices. Loop-closure will get extra favour from continuous accumulating data from k8s automatically.
