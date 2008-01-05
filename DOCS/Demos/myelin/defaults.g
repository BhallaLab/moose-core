function settab2const(gate, table, imin, imax, value)
    str gate
    str table
    int i, imin, imax
    float value
    for (i = (imin); i <= (imax); i = i + 1)
        setfield {gate} {table}->table[{i}] {value}
    end
end
