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
We're the day trippers and We've been working on this idea for approximately 8 days a week. 
But at this point, we decided to Let it Be and present our work.

Our idea was to take concepts from the lottery ticket hypothesis and apply them 
to a variational quantum eigensolver. A lot of machine learning algorithms 
over-parameterize. Does a VQE circuit also over-parameterize? Can we prune away 
gates that are doing nothing (approximately)?

### proof of concept
Want $N=12$ and nlayers < 200 with energy (relative) < 1.8%. 
or N=14, 16$ with any number of layers 
