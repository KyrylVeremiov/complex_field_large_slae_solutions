import matplotlib
import matplotlib.pyplot as plt
import os
from itertools import groupby, permutations
from contextlib import redirect_stdout
from operator import itemgetter, irshift

import numpy as np
import scipy
from PIL.ImageColor import colormap

from constants import *
COLORS=list(matplotlib.colors.XKCD_COLORS)

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
    return results

def make_plot_data(results):
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



def draw_results_changing_n(plot_data, logscale=True,treshold=True,subdirectory="",linestyle='solid',linewidth='1'):
    color_index=0

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
                            plt.plot(x,y, label=item["method"]+" with "+item["preconditioner"],
                                     marker='o',markevery=[0], color=COLORS[color_index],
                                     linestyle=linestyle,linewidth=linewidth)
                            color_index+=1
                            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            max_n=max(max_n,max(list(tmp_dict.keys())))
                            # print(list(tmp_dict.keys()))
                if treshold and prop == "residual_norm":
                    plt.plot([0,max_n],[RESIDUAL_NORM_TRESHOLD,RESIDUAL_NORM_TRESHOLD],"red",label="Treshold")
                if logscale:
                    plt.yscale("symlog")
                plt.title(f"Dependence of {title} on {mat_type} matrix size")
                plt.xlabel("n")
                plt.ylabel(title)
                plt.legend(fontsize="9",loc='center left', bbox_to_anchor=(1, 0.5))
                directory_to_save = ANALYSE_DIRECTORY + "/" + f"{mat_type}_matrix" + f"{'/' + subdirectory if subdirectory != '' else ''}"
                check_directory(directory_to_save)
                plt.savefig(directory_to_save + "/" + title + f"{'_logscale' if logscale else ''}",
                            bbox_inches="tight")
                plt.show()


def group_by_n_and_type(data):
    n_dict={}
    for item in data:
        data = item["data"]
        for el in data:
            key=(el["n"],item['matrix_type'])
            if key not in n_dict:
                n_dict[key]=[]
            n_dict[key].append({"name":item["method"]+" with "+item["preconditioner"],
                                    "time":el["time"],"residual_norm":el["residual_norm"],"num_of_iterations":el["num_of_iterations"]})
    return n_dict

def ungroup_by_n_and_type(data):
    new_result_list=[]
    for item in data:
        for el in data[item]:
            # print(item)
            res={}
            res["time"]=el["time"]
            res["residual_norm"]=el["residual_norm"]
            res["num_of_iterations"]=el["num_of_iterations"]
            res["method"]=el["name"].split(' with ')[0]
            res["preconditioner"]=el["name"].split(' with ')[1]
            res["n"]=item[0]
            res["matrix_type"]=item[1]
            new_result_list.append(res)
    return new_result_list

def draw_results_static_n(plot_data, logscale_time=True,logscale_residual=True,
                          treshold=True,grooped=False,subdirectory=""):
    if grooped:
        n_type_dict = plot_data
    else:
        n_type_dict=group_by_n_and_type(plot_data)
    color_index=0

    for item in n_type_dict:
        # fig, ax = plt.subplots()
        if logscale_time:
            plt.xscale("symlog")
            # if min_time==0:
            #     time_shift=min_next_time*0.01
        if logscale_residual:
            plt.yscale("symlog")
            # if min_res==0:
            #     residual_shift=min_next_res*0.01
        max_time = 0

        for el in n_type_dict[item]:
            # print(n,el)
            # print(max_time,el["coord"][0],el["coord"][1],el["name"],type(el["coord"][1]))
            # print(max_time)
            # if not (el["coord"][1]==np.inf and FILTER_NOT_CONVERGENCE):
            max_time = max(max_time, el["time"])
            # if el["name"]== "richardson with gamg":
            #     plt.plot([float(el["coord"][0]),1,2,3],[float(el["coord"][1]),1,2,0],'o')
            #     print(max_time, el["coord"][0], el["coord"][1], el["name"], type(el["coord"][1]),type(el["coord"][0]))
            # else:
            plt.plot(el["time"],el["residual_norm"],'o',label=el["name"],color=COLORS[color_index])
            color_index+=1
            # ax.plot(el["coord"][0],el["coord"][1],'o',label=el["name"])
        # print(max_time)
        if treshold:
            plt.plot([0,max_time],[RESIDUAL_NORM_TRESHOLD,RESIDUAL_NORM_TRESHOLD],"red",label="Treshold")
        # ax.set_xticks((0,max_time))
        plt.legend(fontsize="9",loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"{item[1]} matrix of size n={item[0]}")
        plt.xlabel("time")
        # plt.xlim(0, max_time)
        plt.ylabel("residual norm")

        directory_to_save= ANALYSE_DIRECTORY + "/" + f"{item[1]}_matrix" + f"{'/' + subdirectory if subdirectory != '' else ''}"
        check_directory(directory_to_save)
        plt.savefig(directory_to_save+"/" f"n_{item[0]}_time_residual" + f"{'_logscale' if logscale_residual else ''}",
                    bbox_inches="tight")
        plt.show()

def check_directory(directory_to_save):
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save)


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

def print_with_filter_or_not():
    print(f"{'With' if FILTER_NOT_CONVERGENCE else 'Without'} filtering out non-convergence results")


def print_plot_data(result):
    with (open(ANALYSE_DIRECTORY + "/" + RESULT_FILENAME + ".txt", 'w') as f, redirect_stdout(f)):
        print_with_filter_or_not()
        for item in result:
            print(item["method"] +" with "+ item["preconditioner"]+ " on " +item["matrix_type"])
            for el in item["data"]:
                print(el)


def print_sorted(data, prop, subname=""):
    if prop == "time":
        filename = RESULT_TIME_FILENAME
    elif prop == "num_of_iterations":
        filename = RESULT_ITERATIONS_FILENAME
    elif prop=="residual_norm":
        filename =RESULT_RESIDLUAL_FILENAME
    directory_to_save= ANALYSE_DIRECTORY+'/'
    check_directory(directory_to_save)
    with (open(directory_to_save +f"{subname if subname != '' else ''}"+ filename + ".txt", 'w') as f, redirect_stdout(f)):
        print_with_filter_or_not()
        for item in data.items():
            print(item[0])
            i = 1
            for el in item[1]:
                print(str(i)+": ",el)
                i+=1


def sort_best_methods(result,prop,number_to_get=-1, grooped=False):
    sorted_n_type_dict={}
    if grooped:
        n_type_dict=result
    else:
        n_type_dict = group_by_n_and_type(result)
    # n_type_list=[[k, v] for k, v in sorted(n_type_dict.items(), key=lambda item: item["time"])]
    # print(n_type_dict.items())
    for item in n_type_dict.items():
        sorted_list=item[1]
        sorted_list.sort(key=lambda el: el[prop])
        if number_to_get<0:
            sorted_n_type_dict[item[0]]=sorted_list
        else:
            sorted_n_type_dict[item[0]]=sorted_list[:number_to_get]
    return sorted_n_type_dict

def get_names_set(data):
    names_set=set()
    for item in data:
        names_set.add((item["method"],item["preconditioner"]))
    return names_set

def get_data_from_names_set(data,names_set):
    chosen_data=[]
    for item in data:
        if (item["method"],item["preconditioner"]) in names_set:
            chosen_data.append(item)
    return chosen_data

def main():
        results=get_results()
        plot_data=make_plot_data(results)
        if FILTER_NOT_CONVERGENCE:
            plot_data=filter_non_convergence(plot_data)
        print_plot_data(plot_data)
        draw_results_changing_n(plot_data, logscale=True,treshold=True,linestyle='solid',linewidth=0.7)
        draw_results_static_n(plot_data,logscale_time=False,logscale_residual=True,treshold=True)

        # for prop in ["time", "num_of_iterations", "residual_norm"]:
        #     sorted_result = sort_best_methods(plot_data, prop)
        #     print_sorted(sorted_result, prop)

        num_to_get=5
        for prop in ["time", "num_of_iterations", "residual_norm"]:
            sorted_result = sort_best_methods(plot_data, prop,number_to_get=num_to_get)
            subdirectory=f"best_{str(num_to_get)}_{prop}"
            print_sorted(sorted_result, prop, subname=f"best_{str(num_to_get)}_")
            # draw_results_static_n(sorted_result,logscale_time=False,logscale_residual=True,
            #                       treshold=False,grooped=True,
            #                       subdirectory=subdirectory)

            ungrooped_sorted_result=ungroup_by_n_and_type(sorted_result)
            names_set=get_names_set(ungrooped_sorted_result)
            new_plot_data=get_data_from_names_set(plot_data,names_set)
            # new_plot_data=make_plot_data(ungrooped_sorted_result)
            draw_results_changing_n(new_plot_data, logscale=True,treshold=False,subdirectory=subdirectory)



if __name__ == "__main__":
    main()