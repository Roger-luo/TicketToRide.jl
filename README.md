# TicketToRide

[![Build Status](https://travis-ci.com/Roger-luo/TicketToRide.jl.svg?branch=master)](https://travis-ci.com/Roger-luo/TicketToRide.jl)
[![Codecov](https://codecov.io/gh/Roger-luo/TicketToRide.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Roger-luo/TicketToRide.jl)

## Installation

in Julia REPL, press `]` then type the following command

```jl
dev https://github.com/Roger-luo/TicketToRide.jl.git
```

or just use `git`

```jl
git clone https://github.com/Roger-luo/TicketToRide.jl.git
```

## Lyrics for QHACK presentation

(*Sing!*)


I think I'm gonna be sad


QHACK's done today


The code that's driving us mad


Is done halfway (not really though)



Ahh we've got a ticket to ride


We've got a ticket to ride


We've got a ticket to ride


But Xanadu don't care


(*Fin*)

Hello, hello, hello!

It's wonderful to be here, it's certainly a thrill! We have such a lovely audience.

It's great to could come together, right, over at Xanadu. 

We're The DayTrippers, and we've stend the Last Hard day's Night working on our (Quantum Lottery) Ticket to Ride project.

It's been a Long and Winding road, and we could easily work on this for the 8 Days this Week, but we have to Let it be and present our work. 

with a Little Help from our Friends at Xanadu, we

Our idea was to take concepts from the lottery ticket hypothesis and apply them 
to a variational quantum eigensolver. A lot of machine learning algorithms 
over-parameterize. Does a VQE circuit also over-parameterize? Can we prune away 
gates that are doing nothing (approximately)?

### proof of concept
Want $N=12$ and nlayers < 200 with energy (relative) < 1.8%. 
or N=14, 16$ with any number of layers 
