# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import tkinter as tk

import smo_simple_study as SMOsimple
import smo_complete_study as SMOcomplete
import non_linear_smo_study as SMOnonLinear

window = tk.Tk()
window.title('SMO 模式选择')
window.geometry('300x200')

var = tk.StringVar()
l = tk.Label(window, bg='yellow', width=20, text='请选择 SMO 模式')
l.pack()

SMO_mode = ''

def print_selection():
    if var.get() == 'A':
        l.config(text=var.get()+'、简单SMO模式 ')
    elif var.get() == 'B':
        l.config(text=var.get() + '、复杂SMO模式 ')
    else:
        l.config(text=var.get() + '、非线性SMO模式 ')
    SMO_mode = var.get()
    print("SMO_mode: ",SMO_mode)

r1 = tk.Radiobutton(window, text='A、简单SMO模式',
                    variable=var, value='A',
                    command=print_selection)
r1.pack()
r2 = tk.Radiobutton(window, text='B、复杂SMO模式',
                    variable=var, value='B',
                    command=print_selection)
r2.pack()
r3 = tk.Radiobutton(window, text='C、非线性SMO模式',
                    variable=var, value='C',
                    command=print_selection)
r3.pack()


def summit_mode():
    print("var.get(): ", var.get())
    SMO_mode = var.get()
    if SMO_mode == 'A':
        print("a")
        SMOsimple.main()
    elif SMO_mode == 'B':
        print("b")
        SMOcomplete.main()
    elif SMO_mode == 'C':
        print("c")
        SMOnonLinear.testRbf()
    else:
        print("no change")

btn_login = tk.Button(window, text='确定', command=summit_mode)#定义一个`button`按钮，名为`Login`,触发命令为`usr_login`
btn_login.place(x=120, y=150)



window.mainloop()