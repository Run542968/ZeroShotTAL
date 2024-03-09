#### 分支："support_DeformableDETR"，这个分支重新构造了程序
- [x] base: 标准的DETR
- [x] base + actionness: 增加一个单独的actoinness head，预测类别无关的前景分数
- [x] base + actionness + distillation: 再增加一个蒸馏loss，蒸馏的位置是经过class_emb()后得到的embedding，这个embedding既用于蒸馏，又用于计算分类CE loss
- [x] base + actionness + distillation + salient: 再增加一个在transformer encoder出来的特征那里，增加前背景区分的masking loss（这里命名是salient，懒得改了） 

#### 20240309
- [x] 添加--enable_element
  - 构造K个可学习的embedding作为dynamic element，经过class_embed的query特征通过cross-attention聚合这些element，然后residual（带权衡系数）连接以后再送去进行CE loss
  - 设计初衷：在base类上学习一些comment的dynamic元素，使得appearance相似的query具有可区分性
- [ ] 添加--compact_loss
  - 给actionness head之前加一个feature projection，然后对project后的特征加一个batch内一致性的约束，目的是增强前景特征的表达能力，从而提高对于位置动作的定位能力
  - 这个loss是搭配actionness loss一起用的