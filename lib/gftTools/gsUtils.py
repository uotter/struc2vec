# -*- coding: utf-8 -*-
from lib.gftTools import gftIO
import numpy as np
import pandas as pd
import xarray

def getCashGid():
    return gftIO.strSet2Np(np.array(['0AC062D610A1481FA5561EC286146BCC']))


def getGodGid():
    return np.chararray(1, itemsize=16, buffer='\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')


def getSingleGodGid():
    iv = 0
    return iv.to_bytes(16,byteorder='little')

def getParm(dict_parm, parmName, defaultValue):
    if parmName in dict_parm:
        return dict_parm[parmName]
    else:
        return defaultValue


def alignDate(sourceDates, targetDates, method='ffill', returnidx=False):
    """
    source dates: rebalance dates
    target dates: all dates
    """

    df_dateMap = pd.DataFrame({'targetDate':targetDates, 'idx':np.arange(len(targetDates))}, index=targetDates)
    if isinstance(sourceDates, pd.Timestamp):
        df_dateMap = df_dateMap.reindex([sourceDates], method=method)
    else:
        df_dateMap = df_dateMap.reindex(sourceDates, method=method)
    if returnidx:
        result = np.array(df_dateMap['idx'])
        if isinstance(sourceDates, pd.Timestamp):
            result = result[0]
    else:
        if isinstance(sourceDates, pd.Timestamp):
            result = df_dateMap['targetDate'][0]
        else:
            if isinstance(sourceDates, pd.DatetimeIndex):
                result = pd.DatetimeIndex(df_dateMap['targetDate'])
            else:
                result = np.array(df_dateMap['targetDate'])
    return result


def roundToLot(sizeArray, lotSize):
    if lotSize > 0:
        sizeArray = sizeArray.fillna(0)
        return np.sign(sizeArray) * lotSize * np.floor(np.round(abs(sizeArray)) / max(1., lotSize))
    else:
        return sizeArray


def cut2bin(ser_x, totalBinNum, ascending=False):
    # calculate bin size
    totalBinNum = int(totalBinNum)
    xlen = len(ser_x)
    arr_binsize = np.repeat(xlen // totalBinNum, totalBinNum)
    remaining = xlen % totalBinNum
    if remaining > 0:
        arr_binsize[:remaining] += 1
    # map each signal to its binIdx
    arr_binmap = np.repeat(np.arange(totalBinNum) + 1, arr_binsize)
    ser_xrank = ser_x.rank(method='first', ascending=ascending)
    ser_result = pd.Series(
        arr_binmap[np.array(ser_xrank.values - 1, dtype='int')])
    return ser_result


class Strategy:
    #hold the strategy result, including cumret, holding
    def __init__(self, gid, cumret, holding):
        # save matrix and column table.
        self.gid = gid      
        self.cumret = cumret
        self.holding = holding
        self.type = "Strategy"
        self.cumretGid = None
        self.holdingGid = None




def isGodColumns(col):
    if col.size == 1:
        return col[0].__len__() == 0
    return False

class InputOperation:
    def __init__(self, just_do, left_value, right_value, order_changed):
        self.just_do = just_do
        self.left_value = left_value.input
        self.right_value = right_value.input
        self.order_changed = order_changed


class matrixCommonInfo:
    def __init__(self, rawInput):
        if isinstance(rawInput, gftIO.GftTable):
            rawInput = rawInput.asMatrix()
            self.is_df = True
        elif isinstance(rawInput, pd.DataFrame):
            self.is_df = True
        else:
            self.is_df = False

        if self.is_df:
            self.is_nonsymbol = isGodColumns(rawInput.columns)
            if self.is_nonsymbol:
                rawInput = rawInput[rawInput.columns[0]]
        else:
            self.is_nonsymbol = False
        self.input = rawInput

    def align_matrix(self, common_index, common_columns):
        if self.is_df:
            if common_index is not None:
                self.input = self.input.reindex(common_index)
            if not self.is_nonsymbol and common_columns is not None:
                if common_columns is not None:
                    self.input = self.input[common_columns]

    def get_operation(self, another):
        if self.is_nonsymbol:
            if (another.is_df and not another.is_nonsymbol):
                return InputOperation(False, another, self, True)
        elif self.is_df:
            if another.is_nonsymbol:
                return InputOperation(False, self, another, False)

        return InputOperation(True, self, another, False)


def merge_info_inplace(info_list):
    common_index = None
    common_columns = None
    for info in info_list:
        if info.is_df:
            if common_index is None:
                common_index = info.input.index
            else:
                common_index = np.intersect1d(common_index, info.input.index)
            if not info.is_nonsymbol:
                if common_columns is None:
                    common_columns = info.input.columns
                else:
                    common_columns = np.intersect1d(common_columns, info.input.columns)

    if (common_index is not None) or (common_columns is not None):
        for info in info_list:
            info.align_matrix(common_index, common_columns)

    return info_list


def align_input(*inputs):
    input_list = []
    for input in inputs:
        input_list.append(matrixCommonInfo(input))
    return merge_info_inplace(input_list)

def classify(context, df_x,df_y,winSize,winStep, clf):
    # bunch of scoresssss
    from lib.gftTools import gsUtils
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    '''
    -------- parameter----------
    DataFrame:{columns=["date","x1","x2", ..., "xn"]}
    DataFrame:{columns=["date","y"]}
    winSize: float
    winSteop: float
    
    ---------return----------
    DataFrame:{columns=["date","y"]}
    
    assumption:
    1. 'xi' has been sorted by 'date'
    2. 'y' cloumn in 'X0' has been shifted
        
    '''

    if isinstance(df_x, gftIO.GftTable):
        df_x = df_x.asColumnTab()
    if isinstance(df_y,dict):
        df_y = df_y["y"]
    if isinstance(df_y, gftIO.GftTable):
        df_y = df_y.asColumnTab()
    
    # convert parameter type
    winSize = int(winSize)
    winStep = int(winStep) 
    
    # NOTICE: integer will be regraged as O by GS, but classifier need int
    value_column = gsUtils.find_value_column(df_y)  # value_column: value
                                                   # df_y.columns:Index(['date', 'value'], dtype='object')
    df_y.rename(columns={value_column:"y"},inplace=True)
    df_y.y=df_y.y.astype(int)
    # change column name
    for col_name in df_y.columns:
        if isinstance(df_y.ix[0,col_name],pd.Timestamp):
            df_y.rename(columns={col_name:"date"},inplace=True)
            break
    # remove meanless columns
    df_y=df_y[["date","y"]]
    
    # merge data
    df_x = df_x.sort_values("date",ascending=True)
    df_y = df_y.sort_values("date",ascending=True)
    df_y = df_y.set_index(np.arange(len(df_y))) # indentify index: start from 0

    # frequency: if y_freq > x_freq, slice dates
    dates_x = df_x['date']
    dates_y = df_y['date']
    inters_dates = pd.Series(np.intersect1d(pd.Series(dates_x), pd.Series(dates_y)))
    union_dates =  np.unique(np.union1d(dates_x, dates_y))
    if len(inters_dates)==0:
        raise ValueError('No common dates between x and y!')
    if len(inters_dates) != len(union_dates):
        alldates = inters_dates
        df_x = df_x[df_x['date'].isin(alldates)]
        df_y = df_y[df_y['date'].isin(alldates)]
    df_x=df_x.reindex_axis(sorted(df_x.columns), axis=1)
    df_y=df_y.reindex_axis(sorted(df_y.columns), axis=1)
    # data to be trained
    df_data=pd.merge_ordered(df_x,df_y,on="date",how="outer") 

    # value check
    if len(df_data.index) < winSize + 1:
        raise ValueError("the number of input data is not enough")
    
    # rolling
    ls_predicted=[]
    for i in range(len(df_data.index)):
        if i<winSize:
            ls_predicted+=[np.nan]
        else:
            start_index=i-winSize
            # fit
            n_x_train= df_data.iloc[start_index:i,1:-1].values
            n_y_train= df_data.iloc[start_index:i,-1].values
            clf.fit(n_x_train, n_y_train)
            # predict
            n_x_test = df_data.iloc[[i],1:-1]
            y_test = clf.predict(n_x_test)[0]
            ls_predicted += [y_test]
    
    df_data["predicted"]=ls_predicted
    #print(ls_predicted)
    
    # drop na
    df_data=df_data.dropna()
    #print(df_data)
    
    # scoressssssss
    y_true=df_data["y"].astype(int)
    y_pred=df_data["predicted"].astype(int)
    num_accuracy_score=accuracy_score(y_true,y_pred)
    #print("accuracy_score:",num_accuracy_score)
    num_f1_score=f1_score(y_true,y_pred,average='macro') # micor, weighted, None
    #print("f1_score:",num_f1_score)
    num_precision_score=precision_score(y_true, y_pred, average='macro') # micor, weighted, None
    #print("precision_score:",num_precision_score)
    num_recall_score=recall_score(y_true, y_pred, average='macro') # micor, weighted, None
    #print("recall_score:",num_recall_score)
    dict_score={"accuracy_score":num_accuracy_score, "f1_score":num_f1_score,"precision_score":num_precision_score, "recall_score":num_recall_score}
    
    # score
    y_test = df_data["predicted"].values
    X_test = df_data.iloc[:,1:-2].values
    num_mean_accuracy=clf.score(X_test , y_test)
    #print(num_score)    
    
    '''
    # feature_importances
    ls_fitness=list(zip(df_data.iloc[:,1:-1],clf.feature_importances_))
    n_fitness=np.array(list(map(list,ls_fitness)))
    df_fitness=pd.DataFrame({"feature":n_fitness[:,0],"importance":n_fitness[:,1]})
    #print(df_fitness)    
    '''
    
    # result
    df_data=df_data[["date","predicted"]]
    #print(df_data)
    
    
    ##y_true and y_predict calculate R square
    y_mean=y_true.mean()
    
    rsquare=((y_pred-y_mean)**2).sum()/((y_true-y_mean)**2).sum()
    
    ##add match pair
    y_true_binary=y_true.apply(lambda x:1 if x >=0 else 0)
    y_pred_binary=y_pred.apply(lambda x:1 if x >=0 else 0)    
    tf=(y_true_binary == y_pred_binary)
    match_pair_pct =len(tf[tf == True])/len(y_true_binary)
    #add max_depth
    max_depth = [estimator.tree_.max_depth for estimator in clf.estimators_]
    max_depth_as_dataframe = pd.DataFrame(max_depth)
    for col_name in max_depth_as_dataframe.columns:
        max_depth_as_dataframe[col_name] = max_depth_as_dataframe[col_name].astype(float)

    max_depth_as_dataframe.columns = ['max_depth']

    dict_result = {"result":df_data,"mean_accuracy":num_mean_accuracy, "scores":dict_score ,"rsquare":rsquare,"match_pair_pct":match_pair_pct,"max_depth":max_depth_as_dataframe} #,"fitness":df_fitness}
    #print(dict_result)
    return dict_result


def _findValueColumn(ls_columns):
    raise ValueError("Do not use this def, use find_value_column(data) instead.")

def find_value_column(data):
    for col_name in data.columns:
        if gftIO.get_column_type(data, col_name) == gftIO.PARAMETER_TYPE_NUMBER_NUMRIC:
            return col_name
    raise ValueError("Value Column is not found in {}!".format(data.columns.tolist()))

def find_date_column(data):
    for col_name in data.columns:
        if gftIO.get_column_type(data, col_name) == gftIO.PARAMETER_TYPE_TIMESTAMP:
            return col_name
    raise ValueError("Date Column isnot found in {}!".format(data.columns.tolist()))

class ExtractDictModelData(object):
    """ model data extraction and getting attribute. """
    def __init__(self, model):
        self.model = model

    def get_input_factor(self, oset_idx):
        """ Get oset idx from risk model.
        Keyword Arguments:
        oset_idx: list of oset gid
        """
        if len(oset_idx) < 1:
            return None
        date_index = self.model.get(oset_idx[0], None).asMatrix().index
        ls_factor_b_char = gftIO.strSet2Np(np.array(oset_idx))
        factor_data = pd.Panel({ls_factor_b_char[key]: self.model.get(factor).asMatrix() for key, factor in enumerate(oset_idx)})

        return factor_data.transpose(1, 2, 0)

    def get_output(self, post_fix, oset_idx=None):
        """ get target data from model

        Keyword Arguments:
        oset_idx: list of oset gid
        poset_fix: 'specificRisk', 'ret_cov', '*.ret'
        """
        if oset_idx is None:
            return self.model.get(post_fix, None)
        else:
            factors_output = pd.DataFrame(
                index=self.model[oset_idx[0]+post_fix].index, columns=oset_idx)
            for value in oset_idx:
                factors_output[value] = self.model[value+post_fix]
            factors_output.columns = gftIO.strSet2Np(
                factors_output.columns.values)
            return factors_output



def merge_matrix(old_data, new_data, old_desc, new_desc):
    if new_desc.required_begin_t <= old_desc.required_begin_t:
        return new_data
    if new_desc.required_begin_t <= old_desc.required_end_t:
        # so slice old data
        old_data = old_data[old_data.index < new_desc.required_begin_t]
    #concat old data with new data
    # concat along index, and use outer join for columns.
    return pd.concat(objs=[old_data, new_data],axis=0,join='outer')




def merge_col_tab(old_data, new_data, old_desc, new_desc):
    print ("merge coltabs, size:{0} and {1}".format(str(old_data.shape), str(new_data.shape)))
    if new_desc.required_begin_t <= old_desc.required_begin_t:
        return new_data

    sorted_old_cols = old_data.columns.sort_values()
    sorted_new_cols = new_data.columns.sort_values()
    if not sorted_old_cols.equals(sorted_new_cols):
        raise Exception("New data's columns{0} is not the same as old data's columns{1}".format(str(sorted_new_cols), str(sorted_old_cols)))

    #bcs pt_name may not right from desc
    pt_name = gftIO.get_pt_name(old_data, old_desc.get_pt_name())
    if pt_name is None:
        return new_data

    if new_desc.required_begin_t <= old_desc.required_end_t:
        # so slice old data
        old_data = old_data[old_data[pt_name] < new_desc.required_begin_t]

    # concat old data with new data
    ret = pd.concat(objs=[old_data, new_data],axis=0,join='outer')
    print("Concated table size:{0}".format(str(ret.shape)))
    return ret


def merge_xarray(old_data, new_data, old_desc, new_desc):
    raise Exception("Not supported yet.")


def merge_data(old_data, new_data, old_desc, new_desc):
    if type(old_data) != type(new_data):
        raise Exception("Can not merge data of differnt types")
    if isinstance(new_data, dict):
        ret = dict()
        for key, val in dict.items():
            old_value = old_data.get(key)
            if old_value is not None:
                ret[key] = merge_data(old_value, val, old_desc, new_desc)
            else:
                ret[key] = val
    if isinstance(new_data, gftIO.GftTable):
        if (new_data.matrix is not None) and (old_data.matrix is not None):
            merged_mtx = merge_matrix(old_data.matrix, new_data.matrix, old_desc, new_desc)
            new_data.matrix = merged_mtx
            new_data.columnTab = None
            return new_data
        if (new_data.columnTab is not None) and (old_data.columnTab is not None):
            merged_col_tab = merge_col_tab(old_data.columnTab, new_data.columnTab, old_desc, new_desc)
            new_data.columnTab = merged_col_tab
            new_data.matrix = None
            return new_data
        raise Exception("Merge GftTable of different type")
    if isinstance(new_data, pd.DataFrame):
        is_new_data_matrix = gftIO.ismatrix(new_data)
        if is_new_data_matrix != gftIO.ismatrix(old_data):
            raise Exception("Merge dataframe of different shape")
        if is_new_data_matrix:
            return merge_matrix(old_data, new_data, old_desc, new_desc)
        else:
            return merge_col_tab(old_data, new_data, old_desc, new_desc)
    if isinstance(new_data, xarray):
        return merge_xarray(old_data, new_data, old_desc, new_desc)
    return new_data



# all caches would be in this struct. so i can get both data and meta.
class CacheData:
    def __init__(self, type, meta, data):
        self.data = data
        self.meta = meta
        self.type = type

    def copy_and_slice_data_with_begin_date(self, begin_date):
        return self

    def copy_and_slice_data_with_end_date(self, end_data):
        return self

    def copy_and_slice_data_with_begin_end_date(self, begin_date, end_date):
        return self


def dumpAll4Cache(type, meta, data, timestamp, filename):
    cache = CacheData(type, meta, data)
    return gftIO.zdump4CacheSever(cache, timestamp, filename)


import pickle

class DataIterator:
    def __init__(self, source_data):
        self.source_data = source_data

    def has_next(self):
        return False

    def next(self):
        return None

    def key(self):
        return None

    def keys(self):
        return None

    def columns(self):
        return None

    def get_source_data(self):
        return self.source_data

    def reset(self):
        pass

    def size(self):
        return 0


class IndexIterator(DataIterator):
    def __init__(self, source_data):
        self.source_data = source_data
        if isinstance(source_data, gftIO.GftTable):
            self.matrix = source_data.asMatrix()
        elif isinstance(source_data, pd.DataFramea):
            self.matrix = source_data
        self.last_index = self.matrix.index.size - 1
        self.pos = -1

    def has_next(self):
        return self.last_index > self.pos

    def next(self):
        self.pos += 1
        return self.matrix.iloc[self.pos]

    def key(self):
        return self.matrix.index[self.pos]

    def keys(self):
        return self.matrix.index.values

    def get_source_data(self):
        return self.source_data

    def columns(self):
        return self.matrix.columns

    def reset(self):
        self.pos = -1

    def size(self):
        return self.last_index + 1

    def __eq__(self, other):
        return type(self)==type(other) and self.source_data == other.source_data and self.pos == other.pos


class columnIterator(DataIterator):
    def __init__(self, source_data):
        self.source_data = source_data
        if isinstance(source_data, gftIO.GftTable):
            self.matrix = source_data.asMatrix()
        elif isinstance(source_data, pd.DataFramea):
            self.matrix = source_data
        self.pos = -1
        self.last_index = self.matrix.columns.size

    def has_next(self):
        return self.last_index > self.pos

    def next(self):
        self.pos += 1
        return self.matrix[self.matrix.columns[self.pos]]

    def key(self):
        return self.matrix.columns[self.pos]

    def keys(self):
        return self.matrix.columns.values

    def get_source_data(self):
        return self.source_data

    def columes(self):
        return self.matrix.index

    def size(self):
        return self.last_index + 1

    def __eq__(self, other):
        return type(self)==type(other) and self.source_data == other.source_data and self.pos == other.pos


class XarrayIterator(DataIterator):
    def __init__(self, source_data, axis = 'date', column_name = 'symbol', index_name = 'factor'):

        if axis == 'data' or axis is None:
            # find axis with timestamp
            self.axis = axis
        else:
            # confirm axis existed.
            self.axis = axis
        if column_name == 'symbol' or column_name is None:
            self.column_name = 'symbol'
        else:
            self.column_name = column_name
        if index_name == 'factor' or index_name is None:
            self.index_name = 'factor'
        else:
            self.index_name = index_name
        self.last_index = source_data[self.axis].size - 1
        self.pos = -1
        self.source_data = source_data.transpose(self.axis, self.index_name, self.column_name)
        self.axis_idx = self.source_data[self.axis]


    def has_next(self):
        return self.count > self.pos

    def next(self):
        self.pos += 1
        return self.source_data[self.axis_idx[self.pos],:,:]

    def key(self):
        return self.axis_idx[self.pos]

    def keys(self):
        return self.axis_idx.values

    def columns(self):
        return (self.source_data[self.index_name],self.source_data[self.column_name])

    def size(self):
        return self.last_index + 1


class DictIterator(DataIterator):
    def __init__(self, dic_data: dict):
        self.source_data = dic_data
        self.it = dic_data.items().__iter__()
        self.count = dic_data.__len__()
        self.key = None

    def has_next(self):
        return self.count > 0

    def next(self):
        self.count -= 1
        key_value = self.it.__next__()
        self.key = key_value[0]
        return key_value[1]

    def key(self):
        return self.key

    def keys(self):
        return self.source_data.keys()

    def get_source_data(self):
        return self.source_data

    def columes(self):
        return 'value'

    def __eq__(self, other):
        return self.source_data == other.source_data

class ListIterator(DataIterator):
    def __init__(self, list_data):
        self.source_data = list_data
        self.last_index = len(list_data) - 1
        self.pos = -1

    def has_next(self):
        return self.last_index > self.pos

    def next(self):
        self.pos += 1
        return self.source_data[self.pos]

    def key(self):
        return self.pos

    def keys(self):
        return range(self.last_index + 1)

    def get_source_data(self):
        return self.source_data

    def columes(self):
        return 'list_value'

    def __eq__(self, other):
        return self.source_data == other.source_data


def getCacheData(value):
    cache = pickle.loads(value)
    if isinstance(cache, CacheData):
        return cache.type, cache.meta, cache.data
    raise Exception("Cache type is not gsMeta.CacheData")


def slice_redundant_result_and_wrap_gft_table_is_necessary(obj, meta):
    if (meta.required_begin_t > meta.input_begin_t) or (meta.required_end_t < meta.input_end_t):
        # may have redundant date in result.
        print("Slice redundant data in result.")
        obj = gftIO.slice_data_inplace_and_ret(obj, gftIO.get_pt_name(obj, meta.get_pt_name()), meta.required_begin_t, meta.required_end_t)

    return gftIO.wrap_gfttable_dataframe_clean_gid(obj)
    
# from gensim.models import KeyedVectors
#
# # Martin 2017-12-28 This is a temp file position. Should be changed when organize multiple models are planned.
# word2vec_en_wikipedia_trained = KeyedVectors.load_word2vec_format('/mnt/hdfs/cacheServer/aiData/word2Vector/word2vec_en_wikipedia', binary=False)


    