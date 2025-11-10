from datetime import datetime
data={}
with open("log.txt",'r') as f:
    for line in f:
        parts=line.strip().split()
        if len(parts)==3:
            user,timestamp,action=parts
            if user not in data:
                data[user]=[]
            data[user].append((timestamp,action))
max_user=0 
for user , events in data.items():
    total=0
    login_time=None
    for timestamp,action in sorted(events):
        t=datetime.fromisoformat(timestamp)
        if action=='login':
            login_time=t
        elif action=='logout' and login_time:
            total+=(t-login_time).total_seconds()
            login_time=None
    hours=round(total/3600,2)
    if max_user<hours:
        max_user=hours
        name=user
    print(user,hours)

print("maximun logined user:",name,max_user)

