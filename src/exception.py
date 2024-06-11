import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_msg = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_msg

def error_message_detail(error, error_detail:sys):
    _, _, exc = error_detail.exc_info()
    file_name = exc.tb_frame.f_code.co_filename
    line_number = exc.tb_lineno
    message = str(error)
    error_msg = f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{message}]"
    return error_msg