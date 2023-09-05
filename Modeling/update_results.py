import sys
from utils.mail_delivery import catch_results

if __name__ == '__main__':
    for arg in sys.argv:
        if '-p=' in arg: password = arg.split('=')[-1]
        if '-m=' in arg: mail = arg.split('=')[-1]
        if '-t=' in arg: tag = arg.split('=')[-1]
    
    catch_results(mail, password, tag)