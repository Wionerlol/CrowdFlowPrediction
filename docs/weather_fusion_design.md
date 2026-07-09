# AGFormer 调休日历与天气门控融合方案

更新时间：2026-07-06

## 1. 日历修正

项目原有 `data/raw/taxibj/BJ_Holiday.txt` 继续作为法定放假日期来源。
该文件不包含周末补班日，因此新增 2013–2016 年国务院办公厅通知中的补班日期，
用于生成：

- `is_holiday`
- `is_weekend`
- `is_workday`（补班周末覆盖普通周末规则）
- `is_makeup_workday`

通知来源：

- 2013：国家体育总局转载国务院办公厅通知
  https://www.sport.gov.cn/bgt/n4962/c649722/content.html
- 2014：国办发明电〔2013〕28号
  https://www.yueyang.gov.cn/gggs/szbm/content_230742.html
- 2015：国办发明电〔2014〕28号
  https://www.nea.gov.cn/2015-01/08/c_133904166.htm
- 2016：国办发明电〔2015〕18号
  https://www.beijing.gov.cn/zhengce/gwywj/201905/t20190522_58774.html

## 2. 天气对齐审计

数据源：`data/processed/weather_bj_merged.csv`

- 天气行数：48,736
- 时间范围：2013-06-30 16:00 UTC 至 2016-04-10 23:30 UTC
- 时间频率：连续 30 分钟
- TaxiBJ 时间戳：22,484
- 精确匹配：22,484 / 22,484（100%）
- 重复时间戳：0
- 对齐后缺失值：0

天气 CSV 使用带时区的 UTC 字符串，TaxiBJ 使用无时区 `datetime64` 表示 UTC；
对齐时显式将 TaxiBJ 时间戳本地化为 UTC，避免“看似相同但匹配为零”的时区错误。

## 3. 天气特征（18维）

连续特征 11 维：温度、露点、气压、湿度、风速、降水、降雨、降雪、云量、
风向正弦、风向余弦。降水相关变量先 `log1p`，所有连续变量只使用训练集均值和
标准差归一化。

天气类型 7 维：晴、云、雾、毛毛雨、雨、雪、强对流，按 WMO weather code
分组后 one-hot 编码。

## 4. 门控融合

流量、时间、静态和天气分别投影到 `d_model`，上下文拼接并压缩后计算门：

```text
context = MLP([time, static, weather])
gate = sigmoid(Linear([flow, context]))
hidden = flow + gate * context
```

天气不与流量直接无条件相加，模型可以按节点和时间关闭无效天气信息。

## 5. Seed 42 筛选实验

公平比较两组：

1. `Calendar + Time + Static`
2. `Calendar + Time + Static + Historical Weather`

两组使用相同的连续窗口过滤、数据切分、随机种子、模型容量和训练策略。
当前天气仅使用预测起点之前的历史天气，不读取目标窗口真实未来天气，因此无未来
信息泄漏。未来天气预报编码器留作后续独立实验。
