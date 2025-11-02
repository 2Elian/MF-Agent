# step1: 创建网络
docker network create es-net
在Docker中，网络可以用来连接多个容器，让它们能够相互通信。
# step2：拉取es
docker pull elasticsearch:8.7.0

# 启动

docker run -d \
  --name es \
  --network es-net \
  -p 9200:9200 \
  -p 9300:9300 \
  -v /data1/nuist_llm/java-project/elasticsearch/data:/usr/share/elasticsearch/data \
  -v /data1/nuist_llm/java-project/elasticsearch/plugins:/usr/share/elasticsearch/plugins \
  -v /data1/nuist_llm/java-project/elasticsearch/config/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.7.0

# 更改验证
docker cp es:/usr/share/elasticsearch/config/elasticsearch.yml /data1/nuist_llm/java-project/elasticsearch/config

修改xpack.security.enabled: false

复制回容器
docker cp /data1/nuist_llm/java-project/elasticsearch/config/elasticsearch.yml es:/usr/share/elasticsearch/config/

重启
docker restart es


最后http://<your_ip>:9200
