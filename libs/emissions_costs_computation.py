def compute_emissions_pax_short_mid_flights(distance, seats):
    # distance in km
    # seats in the aircraft
    # based on model from https://www.mdpi.com/2071-1050/13/18/10401
    

    if distance < 200:
        distance = 200

    if seats > 190:
        seats = 190
    if seats < 72:
        seats = 72

    if 2500>=distance >= 200 and 72 <= seats <= 190: #Westminster formula
        gco2_seat_ask = 167.8 + (2.153*10**4)/distance - 4.083*10**-2 * distance - 0.679*seats + 2.39*10**-4 * distance * seats
        kco2_seat = gco2_seat_ask * distance / 1000
        return round(kco2_seat, 2)
    if distance >2500: #BHL formula
        kco2_seat=1.8453*distance**(-0.401)*1.9*distance
        return round(kco2_seat,2)
    else:
        return None


def compute_costs_air(distance):
    return round(15.784 * distance**(-0.651) * distance, 2)


def compute_emissions_rail(distance, country):
    coefficient = None
    emissions_rail = None

    if country == 'ED':
        coefficient = 0.029
        #if distance <= 500:
        #    co2 = distance * (0.67 + 0.00067 * distance) * 0.029 + distance * (1 - (0.67 + 0.00067 * distance)) * 0.059104
        #else:
        #    co2 = distance * 0.93 * 0.029 + distance * (1 - 0.93) * 0.059104
    elif country == 'LE':
        coefficient = 0.029

    if coefficient is not None:
        emissions_rail = round(distance * coefficient, 2)

    return emissions_rail


def compute_costs_rail(distance, country):
    cost = None

    if country == 'ED':
        cost = round(0.9254 * distance**(-0.281) * distance, 2)
    elif country == 'LE':
        if distance <= 300:
            cost = round(0.0926 * distance, 2)
        else:
            cost = round(0.1055 * distance, 2)

    return cost
