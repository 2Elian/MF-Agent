# step1:
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/milvusdb/milvus:v2.5.4
# webui
http://172.16.107.15:9091/webui/

# 启动 Milvus 容器
bash standalone_embed.sh start
# 停止 Milvus 容器
bash standalone_embed.sh stop
# 删除 Milvus 容器及数据
bash standalone_embed.sh delete
# 更新 Milvus 服务
$ bash standalone_embed.sh upgrade
# 重启服务
bash standalone_embed.sh restart
