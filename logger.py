import logging
import os
from datetime import datetime

####this will keep adding new folders with logs
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

##Folder names
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

#Folder creation
os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s %(lineno)d %(name)-12s %(levelname)-8s %(message)s',
    level=logging.INFO
)
