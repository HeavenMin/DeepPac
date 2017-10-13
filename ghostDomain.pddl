(define (domain ghostDomain)
  (:requirements :strips :typing :conditional-effects)
  (:types position)
  (:predicates  (location ?curPos - position)
                (adjacent ?curPos ?nextPos - position)
                (scareExist)
                (pacmanLoc ?curPos - position)

  )
  (:action moveWhenScareNotExist
        :parameters (?curPos ?nextPos - position)
        :precondition (and (not (scareExist))
                           (location ?curPos)
                           (adjacent ?curPos ?nextPos)
                           
                       )
        :effect   (and (not (location ?curPos))
                       (location ?nextPos)
                       (when (pacmanLoc ?nextPos)
                             (not (pacmanLoc ?nextPos)))
                  )
  )

  (:action moveWhenScare
        :parameters (?curPos ?nextPos ?pacmanPos - position)
        :precondition (and (scareExist)
                           (location ?curPos)
                           (pacmanLoc ?pacmanPos)
                           (adjacent ?curPos ?nextPos)
                           (not (adjacent ?nextPos ?pacmanPos))
                           (not (pacmanLoc ?nextPos))

                           
                       )
        :effect   (and (not (scareExist))
                       (location ?nextPos)
                       (not (location ?curPos))                     
                   )
  )

  
)