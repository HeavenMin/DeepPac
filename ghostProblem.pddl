(define (problem ghostProblem)
   (:domain ghostDomain)
   (:objects  p_0_0 p_0_1 p_0_2 p_0_3 p_0_4 p_0_5 p_0_6 p_0_7 p_0_8 p_0_9 p_0_10 p_0_11 p_0_12 
			  p_1_0 p_1_1 p_1_2 p_1_3 p_1_4 p_1_5 p_1_6 p_1_7 p_1_8 p_1_9 p_1_10 p_1_11 p_1_12 
			  p_2_0 p_2_1 p_2_2 p_2_3 p_2_4 p_2_5 p_2_6 p_2_7 p_2_8 p_2_9 p_2_10 p_2_11 p_2_12 
			  p_3_0 p_3_1 p_3_2 p_3_3 p_3_4 p_3_5 p_3_6 p_3_7 p_3_8 p_3_9 p_3_10 p_3_11 p_3_12 
			  p_4_0 p_4_1 p_4_2 p_4_3 p_4_4 p_4_5 p_4_6 p_4_7 p_4_8 p_4_9 p_4_10 p_4_11 p_4_12 
	 	      p_5_0 p_5_1 p_5_2 p_5_3 p_5_4 p_5_5 p_5_6 p_5_7 p_5_8 p_5_9 p_5_10 p_5_11  - position)

   (:init 
		(adjacent p_1_1 p_2_1)
		(adjacent p_1_1 p_1_2)
		(adjacent p_1_2 p_2_2)
		(adjacent p_1_2 p_1_1)
		(adjacent p_1_2 p_1_3)
		(adjacent p_1_3 p_2_3)
		(adjacent p_1_3 p_1_2)
		(adjacent p_1_3 p_1_4)
		(adjacent p_1_4 p_2_4)
		(adjacent p_1_4 p_1_3)
		(adjacent p_1_4 p_1_5)
		(adjacent p_1_5 p_2_5)
		(adjacent p_1_5 p_1_4)
		(adjacent p_1_5 p_1_6)
		(adjacent p_1_6 p_2_6)
		(adjacent p_1_6 p_1_5)
		(adjacent p_1_6 p_1_7)
		(adjacent p_1_7 p_1_6)
		(adjacent p_1_7 p_1_8)
		(adjacent p_1_8 p_1_7)
		(adjacent p_1_8 p_1_9)
		(adjacent p_1_9 p_1_8)
		(adjacent p_1_9 p_1_10)
		(adjacent p_1_10 p_1_9)
		(adjacent p_2_1 p_1_1)
		(adjacent p_2_1 p_2_2)
		(adjacent p_2_2 p_1_2)
		(adjacent p_2_2 p_2_1)
		(adjacent p_2_2 p_2_3)
		(adjacent p_2_3 p_1_3)
		(adjacent p_2_3 p_2_2)
		(adjacent p_2_3 p_2_4)
		(adjacent p_2_4 p_1_4)
		(adjacent p_2_4 p_2_3)
		(adjacent p_2_4 p_2_5)
		(adjacent p_2_5 p_1_5)
		(adjacent p_2_5 p_3_5)
		(adjacent p_2_5 p_2_4)
		(adjacent p_2_5 p_2_6)
		(adjacent p_2_6 p_1_6)
		(adjacent p_2_6 p_3_6)
		(adjacent p_2_6 p_2_5)
		(adjacent p_3_5 p_2_5)
		(adjacent p_3_5 p_4_5)
		(adjacent p_3_5 p_3_6)
		(adjacent p_3_6 p_2_6)
		(adjacent p_3_6 p_4_6)
		(adjacent p_3_6 p_3_5)
		(adjacent p_3_6 p_3_7)
		(adjacent p_3_7 p_4_7)
		(adjacent p_3_7 p_3_6)
		(adjacent p_3_7 p_3_8)
		(adjacent p_3_8 p_4_8)
		(adjacent p_3_8 p_3_7)
		(adjacent p_3_8 p_3_9)
		(adjacent p_3_9 p_4_9)
		(adjacent p_3_9 p_3_8)
		(adjacent p_3_9 p_3_10)
		(adjacent p_3_10 p_4_10)
		(adjacent p_3_10 p_3_9)
		(adjacent p_4_1 p_4_2)
		(adjacent p_4_2 p_4_1)
		(adjacent p_4_2 p_4_3)
		(adjacent p_4_3 p_4_2)
		(adjacent p_4_3 p_4_4)
		(adjacent p_4_4 p_4_3)
		(adjacent p_4_4 p_4_5)
		(adjacent p_4_5 p_3_5)
		(adjacent p_4_5 p_4_4)
		(adjacent p_4_5 p_4_6)
		(adjacent p_4_6 p_3_6)
		(adjacent p_4_6 p_4_5)
		(adjacent p_4_6 p_4_7)
		(adjacent p_4_7 p_3_7)
		(adjacent p_4_7 p_4_6)
		(adjacent p_4_7 p_4_8)
		(adjacent p_4_8 p_3_8)
		(adjacent p_4_8 p_4_7)
		(adjacent p_4_8 p_4_9)
		(adjacent p_4_9 p_3_9)
		(adjacent p_4_9 p_4_8)
		(adjacent p_4_9 p_4_10)
		(adjacent p_4_10 p_3_10)
		(adjacent p_4_10 p_4_9)
		(location p_4_1)
		(pacmanLoc p_1_10)
		(scareExist)
	)
   (:goal 
        (and (not (pacmanLoc p_1_10)))
    )
)
   
   
   
   