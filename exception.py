
import sys
import logging
import os
from datetime import datetime
from logger import *

def error_message_detail(error, error_detail:sys):
   
   ###the first argument is the error from Exception class, to get the actual error message from the Exception class
   ##The second argument is the error_detail from sys module that provides the sys object consisting of script name, line number
   _,_,exc_tb =error_detail.exc_info()
   file_name = exc_tb.tb_frame.f_code.co_filename
   error_message = f"Error occurred in python script name {file_name} line number {exc_tb.tb_lineno} error message {error}"
   logging.info(error_message)
   return error_message


class CustomException(Exception):
    def __init__(self, error_message_exception, error_detail:sys):
        super().__init__(error_message_exception)
        self.error_message = error_message_detail(error_message_exception,error_detail=error_detail)

    def __str__(self):
       return self.error_message



# try:
#     #logging.info('Hi')
#     #a=+str(1)
#     a=1/0
# except Exception as e:
#     print(e)
#     raise CustomException(e,sys)