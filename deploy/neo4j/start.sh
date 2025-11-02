sudo docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/MyStrongPassword123 \
  neo4j:5.13
# 访问7474端口进入前端页面
# username = neo4j 
# password = MyStrongPassword123