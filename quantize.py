def quantize(x, num_bits=64):
    if num_bits == 0:
        q_x = x
    else:
        qmin = 0.
        qmax = 2.**num_bits - 1.
        min_val, max_val = x.min(), x.max()

        #print(qmax, qmin,num_bits)
        scale = (max_val - min_val) / (qmax - qmin)
        if scale != 0:
            initial_zero_point = 0
            scale = 1.0        
        else:    
            initial_zero_point = qmin - min_val / scale
        zero_point = 0.0

        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        elif scale == 0:
            zero_point = 0.0        
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)
        q_x = (zero_point + x)*qmax #/ scale
        #q_x.clip(qmin, qmax).round()
        q_x = q_x.clip(qmin, qmax).round()
        
        #print(x[0])    
        #print(min_val,max_val)
        #print(qmin,qmax)
        #print(q_x[0:10])
    
    return q_x
