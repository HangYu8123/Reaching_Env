# import sys, select

# print "You have ten seconds to answer!"

# i, o, e = select.select( [sys.stdin], [], [], 10 )

# if (i):
#   print "You said", sys.stdin.readline().strip()
# else:
#   print "You said nothing!"

# from inputimeout import inputimeout, TimeoutOccurred

# if __name__ == "__main__":
#     c =[]
#     try:
#         c = inputimeout(prompt='hello\n', timeout=3)
#     except TimeoutOccurred:
#         print(c)
#         pass
#     print(c, type(c))

# import time

# sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')

# print('Going to sleep for', sec, 'seconds.')

# time.sleep(int(sec))

# print('Enough of sleeping, I Quit!')
# import signal, readchar
# TIMEOUT = 1 # number of seconds your want for timeout

# def interrupted(signum, frame):
#     "called when read times out"
#     print 'interrupted!'
#     print("^C")


# def input():
#     try:
#             print 'You have 5 seconds to type in your stuff...'
#             foo = readchar.readkey()
#             #print(foo)
#             return True
#     except:
#             # timeout
#             return

# # set alarm
# signal.signal(signal.SIGALRM, interrupted)
# signal.alarm(TIMEOUT)
# s = input()
# if s:
#     print("hhhhhhhhhhhhhhhhhhhhh")
# #print(s)
# # disable the alarm after success
# signal.alarm(0)

