def compute_emissions_pax_short_mid_flights(seats, distance):
    # distance in km
    # seats in the aircraft
    # based on model from https://www.mdpi.com/2071-1050/13/18/10401

    if 2500 >= distance >= 200 and 72 <= seats <= 190:
        gco2_seat_ask = 167.8 + (2.153*10**4)/distance - 4.083*10**-2 * distance - 0.679*seats + 2.39*10**-4 * distance * seats
        gco2_seat = gco2_seat_ask * distance / 1000
        return round(gco2_seat, 2)
    else:
        return None
