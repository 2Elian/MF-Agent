import requests
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import json
import urllib.parse

# ================== 配置区域 ==================
CHECK_INTERVAL = 1800  # 每30分钟检查一次
URL = "https://cpipc.acge.org.cn/contestant/queryMyContest"

# 浏览器抓取的最新 Cookie 和 token
COOKIE = "Hm_lvt_efff57047d75583f6c463eaee32793c4=1762226266,1762393612; HMACCOUNT=30123F7D7995C5A9; JSESSIONID=9FF3197293A749DBF343FF8339CE7940; Hm_lpvt_efff57047d75583f6c463eaee32793c4=1762765352; SERVERID=c23303842b5efd910f0f0e75d3da1262|1762767159|1762765334"
TOKEN = "9FF3197293A749DBF343FF8339CE7940"

# 邮件配置
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
SENDER_EMAIL = "836186855@qq.com"
SENDER_PASSWORD = "ttnamyspbhpsbeae"
RECEIVER_EMAIL = "836186855@qq.com"

STATE_FILE = "latest_award_state.json"
# ============================================

def query_my_contest():
    """POST 请求目标接口，返回 JSON 数据"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://cpipc.acge.org.cn/login/enterMain/contestant/myContest",
        "Origin": "https://cpipc.acge.org.cn",
        "X-Requested-With": "XMLHttpRequest",
        "Cookie": COOKIE,
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
    }

    payload = {
        "conditions": json.dumps({"contestName": None, "participateStatus": None}),
        "_gridInfo": json.dumps({}),
        "pcpServiceVariableOb": json.dumps({}),
        "token": TOKEN
    }

    # 转 URL 编码
    data = urllib.parse.urlencode(payload)

    response = requests.post(URL, headers=headers, data=data)
    response.encoding = "utf-8"

    try:
        return response.json()
    except Exception as e:
        print("❌ 返回非 JSON 数据:", e)
        print(response.text)
        return None

def send_email(subject, body):
    """发送提醒邮件"""
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = Header("竞赛监控系统", "utf-8")
    msg["To"] = Header("用户", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
    print("📩 邮件已发送。")

def check_award_status():
    """检查是否获奖"""
    data = query_my_contest()
    if not data or not isinstance(data, list):
        print("⚠️ 返回数据异常")
        return

    for contest in data:
        contest_name = contest.get("CONTEST_NAME", "")
        award_info = contest.get("AWARD_INFO")

        if "第二十二届中国研究生数学建模竞赛" in contest_name:
            print(f"检测到竞赛：{contest_name}, 当前获奖信息: {award_info}")

            # 读取历史状态
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    last_state = json.load(f)
            except FileNotFoundError:
                last_state = {}

            last_award = last_state.get("AWARD_INFO")

            # 如果AWARD_INFO非空并且有更新 → 发邮件
            if award_info and award_info != last_award:
                subject = f"🏆 获奖更新：{contest_name}"
                body = f"您的比赛结果更新啦！\n\n竞赛名称: {contest_name}\n获奖信息: {award_info}\n\n查看详情：{URL}"
                send_email(subject, body)

                # 保存状态
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump({"AWARD_INFO": award_info}, f, ensure_ascii=False, indent=2)
            else:
                print("暂无更新。")
            break
    else:
        print("未找到目标竞赛。")

if __name__ == "__main__":
    while True:
        try:
            check_award_status()
        except Exception as e:
            print("运行出错:", e)
        time.sleep(CHECK_INTERVAL)
