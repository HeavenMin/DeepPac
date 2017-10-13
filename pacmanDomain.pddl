(define (domain pacmanDomain)
  (:requirements :strips :typing :conditional-effects :equality :fluents)
  (:types position)
  (:predicates  (location ?curPos - position)
                (capsuleLoc ?curPos - position)
                (scareExist)
                (adjacent ?curPos ?nextPos -position)
                (foodLoc ?curPos -position)
                (ghostLoc ?curPos -position)
                
  )

  (:action notScareMove
        :parameters (?curPos ?nextPos ?ghostPos - position)
        :precondition (and (location ?curPos)
                           (adjacent ?curPos ?nextPos)
                           (ghostLoc ?ghostPos)
                           (not (adjacent ?nextPos ?ghostPos))
                           (not (ghostLoc ?nextPos))
                           (not (scareExist))

                      )
        :effect   (and  (location ?nextPos)
                        (not (location ?curPos))
                        (when (foodLoc ?nextPos) (not (foodLoc ?nextPos)))
                        (when (capsuleLoc ?nextPos) (scareExist))
                        (when (capsuleLoc ?nextPos) (not (capsuleLoc ?nextPos)))
                        

                  )
  )

  (:action scareMove
        :parameters (?curPos ?nextPos - position)
        :precondition (and (scareExist)
                           (location ?curPos)
                           (adjacent ?curPos ?nextPos)

                      )
        :effect( and   
                      (location ?nextPos)
                      (not (location ?curPos))
                      (when (foodloc ?nextPos) (not (foodLoc ?nextPos)))
                      (when (ghostLoc ?nextPos) (not (ghostLoc ?nextPos)))          
               )
  )

)