# once-for-all-CenterNet-x
1. Implement CenterNet*/CenterNet2/CenterPoint(PointPillar version) device specialized searched by the once-for-all
2. Design CenterNetF/Probabilistic CenterNetF/CenterPointF(PointPillar version)/Probabilistic CenterPointF(Not sure
   PointPillar version can do this or not) which is a faster architecture inspired by YOLOF.
3. Use 3 phase training designed by DetNAS and reuse the pretrained model from once-for-all as the first phase.
NOTE: PointPillar Version CenterPoint is fast and accurate enough for small-scale point cloud from edge device which
      does not concern about small/sparse object far away.
