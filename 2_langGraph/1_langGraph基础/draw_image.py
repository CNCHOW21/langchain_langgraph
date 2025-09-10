from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='User Request Decomposition and Agent Execution')

# 添加节点
dot.node('1', '用户输入')
dot.node('2', '需求识别')
dot.node('3', '需求拆分')
dot.node('4', 'Agent分配')
dot.node('5', 'Agent A执行')
dot.node('6', 'Agent B执行')
dot.node('7', '结果整合')
dot.node('8', '生成最终回复')
dot.node('9', '输出回复')
dot.node('10', '用户反馈')

# 添加边
dot.edges([('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('4', '6'),
           ('5', '7'), ('6', '7'), ('7', '8'), ('8', '9'), ('9', '10'), ('10', '1')])

# 渲染并保存为文件
dot.render('user_request_decomposition.gv', view=True)
