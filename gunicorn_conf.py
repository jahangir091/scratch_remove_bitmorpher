import os

# Socket path
bind = 'unix:/run/scratch_remove/gunicorn.sock'

# Worker Options
# workers = cpu_count() + 1
workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'

# Logging Options
loglevel = 'debug'
accesslog = '/var/log/scratch_remove/access.log'
errorlog = '/var/log/scratch_remove/error.log'
