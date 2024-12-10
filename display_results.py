import matplotlib.pyplot as plt
import os
from itertools import groupby
from contextlib import redirect_stdout
from operator import itemgetter, irshift

import numpy as np
import scipy

from constants import RESIDUAL_NORM_TRESHOLD,RESULTS_DIRECTORY,DISPLAY_DIRECTORY,FILTER_NOT_CONVERGENCE,RESULT_FILENAME

def get_results():
    results=[]
    filenames = os.listdir(RESULTS_DIRECTORY)
    for filename in filenames:
        file_path=RESULTS_DIRECTORY + "/" + filename
        if os.path.getsize(file_path) != 0:
            res={}
            file=open(RESULTS_DIRECTORY + "/" + filename)
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
                elif line.startswith('matrix type'):  # it get only "n" (with `" "`) at the beginning of line
                    parts = line.split('=')
                    res["matrix_type"] = parts[1].strip(' ')
            results.append(res)

        # prop_set={}
        # for res in results:
        #     for prop in res:
        #         if not(prop in prop_set):
        #             # print(prop)
        #             prop_set[prop]=[]
        #         prop_set[prop].append(res[prop])
    # print(results)
    grouper = itemgetter("method", "preconditioner","matrix_type")
    plot_data = []
    for key, grp in groupby(sorted(results, key=grouper), grouper):
        # plot_data.append(list(grp))
        temp_dict = dict(zip(["method", "preconditioner","matrix_type"], key))
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
            el.pop("matrix_type")
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

# def get_smallest_n(plot_data):
#     return min({item["data"]["n"] for item in plot_data})

def draw_results_changing_n(plot_data, logscale=True,treshold=True):
    special_matrix_types=[el for el in scipy.linalg.__dict__.keys() if el[:1] != '_']
    special_matrix_types.append("random")
    # print(list_matrix_types)
    # smallest_n=get_smallest_n(plot_data)
    matrix_types={item["matrix_type"] for item in plot_data}
    for mat_type in matrix_types:
        if mat_type in special_matrix_types:
            for prop in ["time","num_of_iterations","residual_norm"]:
                title=get_title(prop)
                max_n=0
                for item in plot_data:
                    if item["matrix_type"]==mat_type:
                        data = item["data"]
                        tmp_dict={el["n"]:el[prop] for el in data}
                        tmp_list=sorted(tmp_dict.items())
                        # print(item["matrix_type"],item["method"]+" with "+item["preconditioner"])
                        # print(len(tmp_list))
                        if len(tmp_list)>0:
                            x, y = zip(*tmp_list)
                            # TO NOT SHOW ONE-POINT RESULTS
                            # if len(x)>1:
                            # plt.plot(x,y, label=item["method"]+" with "+item["preconditioner"] +" preconditioner")
                            # print(5)
                            plt.plot(x,y, label=item["method"]+" with "+item["preconditioner"],marker='o',markevery=[0])
                            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            max_n=max(max_n,max(list(tmp_dict.keys())))
                            # print(list(tmp_dict.keys()))
                if treshold and prop == "residual_norm":
                    plt.plot([0,max_n],[RESIDUAL_NORM_TRESHOLD,RESIDUAL_NORM_TRESHOLD],"red",label="Treshold")
                if logscale:
                    plt.yscale("log")
                plt.title(f"Dependence of {title} on {mat_type} matrix size")
                plt.xlabel("n")
                plt.ylabel(title)
                plt.legend(fontsize="9",loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(DISPLAY_DIRECTORY + "/" +f"{mat_type}_matrix_"+ title + f"{'_logscale' if logscale else ''}",
                            bbox_inches="tight")
                plt.show()

def draw_results_static_n(plot_data, logscale_time=True,logscale_residual=True,treshold=True):
    n_dict={}
    for item in plot_data:
        data = item["data"]
        for el in data:
            key=(el["n"],item['matrix_type'])
            if key not in n_dict:
                n_dict[key]=[]
            n_dict[key].append({"name":item["method"]+" with "+item["preconditioner"],
                                    "coord":(el["time"],el["residual_norm"])})
    for item in n_dict:
        max_time=0
        for el in n_dict[item]:
            # print(n,el)
            max_time=max(max_time,el["coord"][0])
            # if not (el["coord"][1]==np.inf and FILTER_NOT_CONVERGENCE):
            plt.plot(el["coord"][0],el["coord"][1],'o',label=el["name"])
        if logscale_time:
            plt.xscale("log")
        if logscale_residual:
            plt.yscale("log")
        # print(max_time)
        if treshold:
            plt.plot([0,max_time],[RESIDUAL_NORM_TRESHOLD,RESIDUAL_NORM_TRESHOLD],"red",label="Treshold")
        plt.legend(fontsize="9",loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"{item[1]} matrix of size n={item[0]}")
        plt.xlabel("time")
        plt.ylabel("residual norm")
        plt.savefig(DISPLAY_DIRECTORY + "/" + f"{item[1]}_matrix_n_{item[0]}_time_residual" + f"{'_logscale' if logscale_residual else ''}",
                    bbox_inches="tight")
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
        print(item["method"] +" with "+ item["preconditioner"]+ " on " +item["matrix_type"])
        for el in item["data"]:
            print(el)

def main():
    with (open(DISPLAY_DIRECTORY + "/" + RESULT_FILENAME + ".txt", 'w') as f, redirect_stdout(f)):
        results=get_results()
        if FILTER_NOT_CONVERGENCE:
            results=filter_non_convergence(results)
        draw_results_changing_n(results, logscale=True,treshold=True)
        draw_results_static_n(results,logscale_time=False,logscale_residual=True,treshold=True)
        print(f"{'With' if FILTER_NOT_CONVERGENCE else 'Without'} filtering out non-convergence results")
        print_results(results)
        # print(type(results[5]["data"][0]["residual_norm")

if __name__ == "__main__":
    main()