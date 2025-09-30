from ultralytics import YOLO, checks, hub
checks()

hub.login('690a8baeae3ada583dc5df02229eebfefbed5e6551')

model = YOLO('https://hub.ultralytics.com/models/fN9lDOjojn7XIb34poRW')
results = model.train()