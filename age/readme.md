#### 年龄分类
- 输入图片是经过目标检测网络检测出来的行人图片，并且经过人工预处理，所有输入图片仅包含行人目标，不包含其他目标。
#### 任务分解
- 样本图片一共包含了7个年龄段的图片，即待处理的问题是7分类问题。实际中，我们将7各类别又重新划分为了4个大类。重新划分是对问题的一个弱化，将多分类问题转化为较少类别的分类问题，因为增加了单类的样本数。
- 任务分类4分类年龄和7分类年龄

four_class_1.0:
	Take the age detection problem as a four class classification problem, 
	and uses standard pytorch library to conduct this task.
four_class:
	Rewrite the ImageFloder class 

