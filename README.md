# CBOW 文本分类项目技术细节

## Vocabulary 类

1. **问题**: 在 Vocabulary 类中，mask_token 对应的索引通过调用 add_token 方法赋值的 self.___属性。  
   **答案**: `self.mask_index`

2. **问题**: lookup_token 方法中，如果 self.mk_index=40，则对未登录词返回 ___.  
   **答案**: `40` 或 `self.mk_index`

3. **问题**: 调用 add_many 方法添加多个 token 时，实际是通过循环调用___方法实现的。  
   **答案**: `add_token`

## CBOWVectorizer 类

4. **问题**: vectorize 方法中，当 vector_length <0 时，最终向源长宽等于 __的长度。  
   **答案**: `原始文本`

5. **问题**: from_dataframe 方法构建到表层，会遍历 Dataframe 中 __和 __两列的内容。  
   **答案**: `text`, `label`

6. **问题**: out_vectorifier(index)的第9列名为 set(&new, vocib, ___.  
   **答案**: `vocab_index`

## CBOWDataset 类

7. **问题**: _max_seq_length 通过计算所有 context 列的 __的最大值得出。  
   **答案**: `长度`

8. **问题**: set_spit 方法通过 self_lookup_dict 选择对应的 __和 __.  
   **答案**: `训练集`, `测试集`

9. **问题**: __getitem__是目的字符串，y_target 通过查找 __列的 token 得到。  
   **答案**: `label`

## 模型结构

10. **问题**: CBOWClassifier 的 forward 中，x_embedded_sum 的计算方式是 embeddings(x_in), __(&len=1)。  
    **答案**: `sum(dim=1)`

11. **问题**: 模型输出层包含 of_out_features 等于 __参数的值。  
    **答案**: `num_classes`

## 训练流程

12. **问题**: generate_batches 函数通过 PyTorch 的 __类实现批量加载。  
    **答案**: `DataLoader`

13. **问题**: 训练时 classifier.sum()的作用是启用 __和 __模式。  
    **答案**: `训练`, `dropout`

14. **问题**: 反向传播前必须执行 __,zero_grad()的空梯度。  
    **答案**: `optimizer`

15. **问题**: compute_accuracy 中 y_pred_values 通过 __方法获取预测类别。  
    **答案**: `torch.max()`

## 训练状态管理

16. **问题**: make_train_state 中 early_stopping_test_val 初始化为 ___.  
    **答案**: `float('inf')`

17. **问题**: update_train_state 在线做 __次验证相关未下调的触发异常。  
    **答案**: `n` (具体次数取决于实现)

18. **问题**: 当验证相关下调时，early_stopping_step 会被重新为 ___.  
    **答案**: `0`

## 设备与随机种子

19. **问题**: set_seed_s everywhere 中与 CUDA 相关的设置是 __,manual_seed_s(leced)。  
    **答案**: `torch.cuda`

20. **问题**: seq.delete 的值很慢 __,st_y_calable()确定。  
    **答案**: `由stable()`

## 推理与测试

21. **问题**: get_closest 函数中排除计算的目标副本总是通过 continue 判断 word = __实现的。  
    **答案**: `target_word`

22. **问题**: 测试程序中的一定要调用 __方法使用 dropout。  
    **答案**: `eval()`

## 关键参数

23. **问题**: CBOWClassifier 的 padding_idx 参数默认值为 ___.  
    **答案**: `0`

24. **问题**: seq 中控制语句最重要的参数是 ___.  
    **答案**: `max_length`

25. **问题**: 学习非阻塞策略 Reduced.RonPatterns 的概念条件是验证损失 __（增加/减少）。  
    **答案**: `增加`
