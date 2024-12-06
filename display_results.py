import matplotlib.pyplot as plt
import os
from itertools import groupby
from contextlib import redirect_stdout
from operator import itemgetter, irshift
from constants import RESIDUAL_NORM_TRESHOLD

DATA_DIRECTORY = "results"
DIRECTORY_TO_SAVE="display"
FILENAME="result"
FILTER_NOT_CONVERGENCE=False


def get_results():
    results=[]
    filenames = os.listdir(DATA_DIRECTORY)
    for filename in filenames:
        res={}
        file=open(DATA_DIRECTORY + "/" + filename)
        for line in file:
            line = line.strip(' \n,')  # remove spaces, tabs but also `,`

            # if 'n' in line:            # it get `n` in any place - ie. 'small n'
            if line.startswith('n='):  # it get only "n" (with `" "`) at the beginning of line
                parts = line.split('=')
                res["n"]=int(parts[1].strip(' '))
            elif line.startswith('method'):  # it get only "n" (with `" "`) at the beginning of line
                parts = line.split('=')
                res["method"]=parts[1].strip(' ')
            elif line.startswith('preconditioner'):  # it get only "n" (with `" "`) at the beginning of line
                parts = line.split('=')
                res["preconditioner"]=parts[1].strip(' ')
            elif line.startswith('time'):  # it get only "n" (with `" "`) at the beginning of line
                parts = line.split('=')
                res["time"]=float(parts[1].split(' ')[0])
            elif line.startswith('residual norm'):  # it get only "n" (with `" "`) at the beginning of line
                parts = line.split('=')
                res["residual_norm"]=float(parts[1].strip(' '))
            elif line.startswith('number of iterations'):  # it get only "n" (with `" "`) at the beginning of line
                parts = line.split('=')
                res["num_of_iterations"]=int(parts[1].strip(' '))
        results.append(res)

        # prop_set={}
        # for res in results:
        #     for prop in res:
        #         if not(prop in prop_set):
        #             # print(prop)
        #             prop_set[prop]=[]
        #         prop_set[prop].append(res[prop])

    grouper = itemgetter("method", "preconditioner")
    plot_data = []
    for key, grp in groupby(sorted(results, key=grouper), grouper):
        # plot_data.append(list(grp))
        temp_dict = dict(zip(["method", "preconditioner"], key))
        # temp_dict["n"] = [item["n"] for item in grp]
        temp_dict["data"] = [item for item in grp]
        # temp_dict["data"] = {item: item for item in grp}
        # for key, grp in groupby(sorted(results, key=grouper), grouper):
        #     temp_dict = dict(zip(["method", "preconditioner"], key))
        #     temp_dict["time"] = [item["time"] for item in grp]
        # for key, grp in groupby(sorted(results, key=grouper), grouper):
        #     temp_dict = dict(zip(["method", "preconditioner"], key))
        #     temp_dict["residual_norm"] = [item["residual_norm"] for item in grp]
        # for key, grp in groupby(sorted(results, key=grouper), grouper):
        #     temp_dict = dict(zip(["method", "preconditioner"], key))
        #     temp_dict["num_of_iterations"] = [item["num_of_iterations"] for item in grp]
        plot_data.append(temp_dict)
    for item in plot_data:
        for el in item["data"]:
            el.pop("method")
            el.pop("preconditioner")
    # print(plot_data)
    return plot_data

def get_title(prop):
    if prop == "time":
        title = "time"
    elif prop == "num_of_iterations":
        title = "number of iterations"
    elif prop == "residual_norm":
        title = "residual norm"
    return title

def draw_results_changing_n(plot_data, logscale=True):
    for prop in ["time","num_of_iterations","residual_norm"]:
        title=get_title(prop)
        for item in plot_data:
            data = item["data"]
            if logscale:
                plt.yscale("log")
            tmp_dict={el["n"]:el[prop] for el in data}
            tmp_list=sorted(tmp_dict.items())
            x, y = zip(*tmp_list)
            # TO NOT SHOW ONE-POINT RESULTS
            # if len(x)>1:
                # plt.plot(x,y, label=item["method"]+" with "+item["preconditioner"] +" preconditioner")
            plt.plot(x,y, label=item["method"]+" with "+item["preconditioner"])
            plt.legend(fontsize="10")
            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"Dependence of {title} on matrix size")
        plt.xlabel("n")
        plt.ylabel(title)
        plt.savefig(DIRECTORY_TO_SAVE+"/"+title)
        plt.show()

def draw_results_static_n(plot_data, logscale_time=True,logscale_residual=True,treshold=True):
    n_dict={}
    for n in plot_data:
        data = n["data"]
        for el in data:
            if el["n"] not in n_dict:
                n_dict[el["n"]]=[]
            n_dict[el["n"]].append({"name":n["method"]+" with "+n["preconditioner"],"coord":(el["time"],el["residual_norm"])})
    for n in n_dict:
        max_time=0
        for el in n_dict[n]:
            # print(n,el)
            max_time=max(max_time,el["coord"][0])
            plt.plot(el["coord"][0],el["coord"][1],'o',label=el["name"])
        if logscale_time:
            plt.xscale("log")
        if logscale_residual:
            plt.yscale("log")
        print(max_time)
        if treshold:
            plt.plot([0,max_time],[RESIDUAL_NORM_TRESHOLD,RESIDUAL_NORM_TRESHOLD],"red",label="Treshold")
        plt.legend(fontsize="10")
        plt.title(f"Matrix size n={n}")
        plt.xlabel("time")
        plt.ylabel("residual norm")
        plt.savefig(DIRECTORY_TO_SAVE+"/"+f"n_{n}_time_residual")
        plt.show()

def filter_non_convergence(result):
    del_res=[]
    for item in result:
        for el in item["data"]:
            # print(el["residual_norm"])
            # print(el)
            # if el["residual_norm"] > 10:
            if el["residual_norm"] > RESIDUAL_NORM_TRESHOLD:
                # item["data"].remove(el)
                del_res.append(el)
                # print(el["residual_norm"])
                # print(plot_data)
    for el in del_res:
        for item in result:
            if el in item["data"]:
                item["data"].remove(el)
    return result

def print_results(result):
    for item in result:
        print(item["method"] +" with "+ item["preconditioner"])
        for el in item["data"]:
            print(el)

def main():
    with (open(DIRECTORY_TO_SAVE+"/"+ FILENAME+".txt", 'w') as f, redirect_stdout(f)):
        results=get_results()
        if FILTER_NOT_CONVERGENCE:
            results=filter_non_convergence(results)
        draw_results_changing_n(results, logscale=True)
        draw_results_static_n(results,logscale_time=False,logscale_residual=True,treshold=True)
        print(f"{'With' if FILTER_NOT_CONVERGENCE else 'Without'} filtering out non-convergence results")
        print_results(results)
        # print(type(results[5]["data"][0]["residual_norm")

if __name__ == "__main__":
    main()