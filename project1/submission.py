########## Question 1 #######################################################################
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):

    def count(hashes_1,hashes_2,offset,alpha_m):
        counter = 0
        i = 0
        while i < len(hashes_1) and counter < alpha_m:
            if abs(hashes_1[i] - hashes_2[i]) <= offset:
                counter = counter + 1
            i = i + 1

        if counter >= alpha_m:
            return True
        else:
            return False

    def binarySearch(low_offset,high_offset):

        while high_offset >= low_offset:
            mid = low_offset + (high_offset - low_offset) // 2
            cand1 = data_hashes.filter(lambda x:count(x[1], query_hashes, mid, alpha_m) != False)
            cand1_count = cand1.count()

            if cand1_count == beta_n:
                return cand1.map(lambda x:x[0])

            elif cand1_count > beta_n:
                high_offset=mid-1

            elif cand1_count < beta_n:
                low_offset=mid+1

        # print(low_offset,mid,high_offset,cand1_count,beta_n)
        if cand1_count >= beta_n:
            return cand1.map(lambda x:x[0])

        else:
            cand1 = data_hashes.filter(lambda x: count(x[1], query_hashes, low_offset, alpha_m) != False)
            return cand1.map(lambda x: x[0])

    offset = 0
    cand = data_hashes.filter(lambda x:count(x[1], query_hashes, offset, alpha_m) != False)
    if cand.count() >= beta_n:
        return cand.map(lambda x:x[0])

    offset = 1
    while True:
        cand = data_hashes.filter(lambda x:count(x[1],query_hashes,offset,alpha_m)!=False)
        # print(cand.collect())
        cand_count = cand.count()

        if cand_count == beta_n:
            return cand.map(lambda x:x[0])

        elif cand_count < beta_n:
            offset = offset*2

        elif cand_count > beta_n:
            return binarySearch(offset//2,offset)



########################################################################################
