import Program

def initial():
    print("\nFor Registering face, Enter 1  \nFor Attendance, Enter 2 \nFor Exit, Enter 0\n")
    a1=input('Give your input : ')
    if a1=='1':
        if Program.Input():
            return initial()
    elif a1=='2':
        if Program.face_rec():
            return initial()
    elif a1=='0':
        print("Thank You! See you Soon...\n")
        exit
    else:
        print("\n#### Please Give The Correct Value ####")
        return initial()

print("\tAttendance")
initial()