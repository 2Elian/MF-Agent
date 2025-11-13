import requests

COOKIE = "Hm_lvt_efff57047d75583f6c463eaee32793c4=1762226266,1762393612; HMACCOUNT=30123F7D7995C5A9; JSESSIONID=9FF3197293A749DBF343FF8339CE7940; Hm_lpvt_efff57047d75583f6c463eaee32793c4=1762765352; SERVERID=c23303842b5efd910f0f0e75d3da1262|1762767159|1762765334"
URL = "https://cpipc.acge.org.cn/contestant/queryMyContest"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://cpipc.acge.org.cn/login/enterMain/contestant/myContest",
    "Origin": "https://cpipc.acge.org.cn",
    "X-Requested-With": "XMLHttpRequest",
    "Cookie": COOKIE,
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
}

res = requests.post(URL, headers=headers, data={})
print(res.text)