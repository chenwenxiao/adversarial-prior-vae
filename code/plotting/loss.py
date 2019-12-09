'''
drawing curves from console log
using curves() to draw from a file
using curves_from_string() to draw from a string text

def curves(kw_list,src_path="./console.log",dst_path="./",dpi=300,seprate=False)

def curves_from_string(kw_list,src,dst_path="./",dpi=300,seprate=False):

kw_list: a list of keyword, 
         every keyword should contain every character of the target

src_path: the path of log file / src: string of log text

dst_path: the directory to save figure

dpi: dpi of figure, default is 300

seperate: save different curve seperately


================*=========================*====================

supporting command line arguments :

    python3 loss.py -f "my_dir/my_console.log" -t "my_plotting_dir/" kw1 kw2 kw3

if kw contains " ", using "+" instead
like: "D+loss" instead of "D loss"  

the picture will save as "my_plotting_dir/loss curves.py"

'''
from matplotlib import pyplot as plt
import re
import sys, os
import getopt

def curves(kw_list,src_path="./console.log",dst_path="./",dpi=300,seprate=False):
    y_min = []
    y_max = []
    fig = plt.figure()
    ax = plt.subplot(111)
    for kw in kw_list:
        pair = curve(kw,src_path=src_path,dst_path=dst_path,dpi=dpi,save=seprate)
        x = pair[0]
        y = pair[1]
        ax.plot(x,y,label=kw)
        y_min.append(min(y))
        y_max.append(max(y))

    if not seprate:
        t_min = min(y_min)
        t_max = max(y_max)
        print(t_min,t_max)
        ax.axis([0,1001,t_min,t_max])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
        plt.tight_layout()
        try:
            os.makedirs(dst_path)
        except BaseException:
            pass
        plt.savefig(dst_path+"loss curves.png",dpi=dpi)

def curve(kw,src_path="./console.log",dst_path="./",dpi=300,save=False):
    name = kw
    x = []
    y = []
    f_cnt = 1
    e_cnt = 1
    with open(src_path) as log:
        for line in log:
            epoch_match = re.search(r"Epoch",line)
            if epoch_match:
                e_cnt += 1
                kw_match = re.search(kw,line)
                if kw_match:
                    result = re.search( r"-?[0-9]+\.?[0-9]*", line[kw_match.end():])
                    if result:
                        f_cnt += 1
                        y.append(float(result.group(0)))
                        x.append(e_cnt)
    log.close()
    y_max = max(y)
    y_min = min(y)
    if save:
        plt.figure()
        plt.plot(x, y)
        print(f"{kw} saving")
        plt.title(kw)
        plt.axis([0,1001,y_min,y_max])
        try:
            os.makedirs(dst_path)
        except BaseException:
            pass
        plt.savefig(dst_path+kw+".png",dpi=dpi)
    print(" info")
    print(f"{kw} found {f_cnt} in {e_cnt} Epoches")
    print(f"ranged from {y_min} to {y_max}")
    return (x,y)

def curves_from_string(kw_list,src,dst_path="./",dpi=300,seprate=False):
    y_min = []
    y_max = []
    fig = plt.figure()
    ax = plt.subplot(111)

    for kw in kw_list:
        pair = curve_from_string(kw,src=src,dst_path=dst_path,dpi=dpi,save=seprate)
        x = pair[0]
        y = pair[1]
        ax.plot(x,y,label=kw)
        y_min.append(min(y))
        y_max.append(max(y))
    if not seprate:
        t_min = min(y_min)
        t_max = max(y_max)
        print(t_min,t_max)
        ax.axis([0,1001,t_min,t_max])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
        plt.tight_layout()
        try:
            os.makedirs(dst_path)
        except BaseException:
            pass
        plt.savefig(dst_path+"loss curves.png",dpi=dpi)

def curve_from_string(kw,src,dst_path="./",dpi=300,save=False):
    name = kw
    x = []
    y = []
    f_cnt = 1
    e_cnt = 1
    log = src.split("Epoch")
    for v_line in log:
        e_cnt += 1
        kw_match = re.search(kw,v_line)
        if kw_match:
            result = re.search( r"-?[0-9]+\.?[0-9]*", v_line[kw_match.end():])
            if result:
                f_cnt += 1
                y.append(float(result.group(0)))
                x.append(e_cnt)

    y_max = max(y)
    y_min = min(y)
    if save:
        plt.figure()
        plt.plot(x, y)
        print(f"{kw} saving")
        plt.title(kw)
        plt.axis([0,1001,y_min,y_max])
        try:
            os.makedirs(dst_path)
        except BaseException:
            pass
        plt.savefig(dst_path+kw+".png",dpi=dpi)
    print(" info")
    print(f"{kw} found {f_cnt} in {e_cnt} Epoches")
    print(f"ranged from {y_min} to {y_max}")
    return (x,y)

def main():
    try:  
        src = ''
        dst = ''
        from_file = None
        opts, args = getopt.getopt(
            sys.argv[1:], "f:s:t:",
            ["file=", "string=", "target="]
            )  
        for opt,arg in opts:
            print(opt,arg)
            if opt == "-s" or opt == "--string" :
                if from_file == None:
                    from_file = False
                    src=arg
                else:
                    return
            elif opt == "-f" or opt == "--file":
                if from_file == None:
                    from_file = True
                    src=arg
                else:
                    return 
            elif opt == "-t" or opt == "--target":
                dst = arg
        new_args =[]
        for word in args:
            new_args.append(word.replace("+"," "))
        if from_file == None:
            print(f"default")
            curves(new_args,"./console.log","./")
        elif from_file:
            print(f"calling curves with kw {new_args}")
            curves(new_args,src,dst)
        else:
            print(f"calling curves string with kw {new_args}")
            curves_from_string(new_args,src,dst)

    except getopt.GetoptError: 
        print("wrong arguments format")

if __name__ == "__main__":
    main()
    # curves(["D loss"],dst_path="./1/")
                