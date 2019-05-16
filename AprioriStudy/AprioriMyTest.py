
def load_data_set():
    """
    Load a sample data set (From Data Mining: Concepts and Techniques, 3th Edition)
    Returns:
        A data set: A list of transactions. Each transaction contains several items.
    """
    print("-" * 5, "load_data_set", "-" * 5)
    data_set = [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]

    print("data_set:",data_set)
    return data_set

def create_C1(data_set):
    """
    产生所有的频繁1-项集
    Create frequent candidate 1-itemset C1 by scaning data set.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
    Returns:
        C1: A set which contains all frequent candidate 1-itemsets
    """
    print("-" * 5, "create_C1", "-" * 5)
    C1 = set()              #用于存储1-项集
    for t in data_set:
        for item in t:
            #rozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。
            item_set = frozenset([item])
            C1.add(item_set)
    #print("C1:",C1)
    return C1

def create_Ck(Lksub1, k):
    """
    创建所有频繁项集
    Create Ck, a set which contains all all frequent candidate k-itemsets
    by Lk-1's own connection operation.
    Args:
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
        k: the item number of a frequent itemset.
    Return:
        Ck: a set which contains all all frequent candidate k-itemsets.
    """
    print("-" * 5, "create_Ck", "-" * 5)
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            #print("-"*10)
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            #print("l1:",l1)
            #print("l1[0:",k-2,"]:",l1[0:k-2])
            l2.sort()
            #print("l2:", l2)
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                #print("Ck_item:",Ck_item)
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck

def is_apriori(Ck_item, Lksub1):
    """
    用于判断待测项集的子集是否为频繁项集，只有子集是频繁子集，待测项集才有可能是频繁项集
    Judge whether a frequent candidate k-itemset satisfy Apriori property.
    Args:
        Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                 candidate k-itemsets.
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    Returns:
        True: satisfying Apriori property.
        False: Not satisfying Apriori property.
    """
    #print("-" * 5, "is_apriori", "-" * 5)
    for item in Ck_item:
        #print("frozenset([item]:",frozenset([item]))
        sub_Ck = Ck_item - frozenset([item])        #通过做差，判断该项集的所有子集是否都为频繁子集
        #print("sub_Ck:", sub_Ck)
        if sub_Ck not in Lksub1:
            return False
    return True

def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    """
    Generate Lk by executing a delete policy from Ck.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        Ck: A set which contains all all frequent candidate k-itemsets.
        min_support: The minimum support.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    Returns:
        Lk: A set which contains all all frequent k-itemsets.
    """
    print("-" * 5, "generate_Lk_by_Ck", "-" * 5)
    Lk = set()                  #存储所有k-频繁项集
    item_count = {}             #统计每一个频繁子集出现的次数
    ##频繁项集计数##
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    #print("item_count:",item_count)
    t_num = float(len(data_set))
    #print("t_num:",t_num)

    ##计算每个频繁项集的支持度##
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    #print("support_data:", support_data)
    #print("Lk:", Lk)
    return Lk

def generate_L(data_set, k, min_support):
    """
    Generate all frequent itemsets.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        k: Maximum number of items for all frequent itemsets.
        min_support: The minimum support.
    Returns:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    """
    print("-"*5,"generate_L","-"*5)
    support_data = {}
    ##计算频繁1-项集##
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    #print("Lksub1:",Lksub1)

    ##计算所有频繁项集##
    L = []                  #存储所有频繁项集
    L.append(Lksub1)
    #print("L:",Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    #print("-" * 10)
    #print("L:",L)
    #print("-" * 10)
    #print("support_data:",support_data)
    return L, support_data

def generate_big_rules(L, support_data, min_conf):
    """
    Generate big rules from frequent itemsets.
    Args:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
        min_conf: Minimal confidence.
    Returns:
        big_rule_list: A list which contains all big rules. Each big rule is represented
                       as a 3-tuple.
    """
    print("-" * 5, "generate_big_rules", "-" * 5)
    print("L:",L)
    print("support_data:",support_data)
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        print("L[",i,"]:",L[i])
        for freq_set in L[i]:
            print("freq_set:",freq_set)
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]            #求条件概率，也就是置信度
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    print("big_rule:",big_rule)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list

if __name__ == "__main__":
    """
    Test
    """
    data_set = load_data_set()
    L, support_data = generate_L(data_set, k=3, min_support=0.2)
    big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)