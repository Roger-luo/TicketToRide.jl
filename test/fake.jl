using TicketToRide
using Test

r = rand_state(10)
@test copy(r) |> put(10, 2=>NoGate()) ≈ r
