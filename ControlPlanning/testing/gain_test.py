import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import longitudinal.gain_scheduler as gs

def manual_sim():
    # myGains = gs.Gains(0.0, 0.0, 0.0, 0.0)
    myGains = gs.Gains(5.0, 5.0, 5.0)
    target =  gs.Gains(1.0, 1.0, 1.0)

    scheduler = gs.GainScheduler(myGains, 0)
    scheduler.start_gain_change(target, 0.0, 5.0)

    scheduler.change_gains(0.5)
    print(f"===== Gains at 0.5 =====")
    scheduler.print_gains()

    scheduler.change_gains(1.0)
    print(f"===== Gains at 1.0 =====")
    scheduler.print_gains()

    scheduler.change_gains(1.5)
    print(f"===== Gains at 1.5 =====")
    scheduler.print_gains()

    scheduler.change_gains(2.0)
    print(f"===== Gains at 2.0 =====")
    scheduler.print_gains()

    scheduler.change_gains(2.5)
    print(f"===== Gains at 2.5 =====")
    scheduler.print_gains()

    scheduler.change_gains(3.0)
    print(f"===== Gains at 3.0 =====")
    scheduler.print_gains()

    scheduler.change_gains(3.5)
    print(f"===== Gains at 3.5 =====")
    scheduler.print_gains()

    scheduler.change_gains(4.0)
    print(f"===== Gains at 4.0 =====")
    scheduler.print_gains()

    scheduler.change_gains(4.5)
    print(f"===== Gains at 4.5 =====")
    scheduler.print_gains()

    scheduler.change_gains(5.0)
    print(f"===== Gains at 5.0 =====")
    scheduler.print_gains()

    scheduler.change_gains(5.5)
    print(f"===== Gains at 5.5 =====")
    scheduler.print_gains()

def simple_transition():
    myGains = gs.Gains(5.0, 5.0, 5.0)
    # target =  gs.Gains(1.0, 1.0, 1.0, 1.0)
    t = 0
    scheduler = gs.GainScheduler(myGains, t)

    scheduler.gain_schedule(0, t)
    while (scheduler.target != None):
        print(f"===== Gains at t {t} =====")
        scheduler.gain_schedule(0, t)
        scheduler.print_gains()
        t += 0.333

    assert scheduler.gains.KP == 0.0
    assert scheduler.gains.KI == 0.0
    assert scheduler.gains.KD == 0.0

    print("Gain transition test passed!")

def change_gains_in_middle_01():
    myGains = gs.Gains(5.0, 5.0, 5.0)
    t = 0
    scheduler = gs.GainScheduler(myGains, t)

    scheduler.gain_schedule(0, t)
    while (scheduler.target != None):
        print(f"===== Gains at t {t} =====")
        if (t > 1.5):
            scheduler.gain_schedule(2, t)
            print(f"End of Ramp Time: {scheduler.t_ramp_end}")
        else:
            scheduler.gain_schedule(0, t)

        scheduler.print_gains()
        t += 0.01
    assert scheduler.gains.KP == 10.0
    assert scheduler.gains.KI == 10.0
    assert scheduler.gains.KD == 10.0

    print("Gain transition test passed!")

def change_gains_in_middle_02():
    myGains = gs.Gains(5.0, 5.0, 5.0)
    t = 0
    scheduler = gs.GainScheduler(myGains, t)

    scheduler.gain_schedule(2, t)
    while (scheduler.target != None):
        print(f"===== Gains at t {t} =====")
        if (t < 1.5):
            scheduler.gain_schedule(2, t)
        else:
            scheduler.gain_schedule(0, t)
            print(f"End of Ramp Time: {scheduler.t_ramp_end}")

        scheduler.print_gains()
        t += 0.01

    
    assert scheduler.gains.KP == 0.0
    assert scheduler.gains.KI == 0.0
    assert scheduler.gains.KD == 0.0

    print("Gain transition test passed!")


def change_gains_in_middle_03():
    myGains = gs.Gains(5.0, 5.0, 5.0)
    t = 0
    scheduler = gs.GainScheduler(myGains, t)

    scheduler.gain_schedule(2, t)
    while (scheduler.target != None):
        print(f"===== Gains at t {t} =====")
        if (t < 1.5):
            scheduler.gain_schedule(2, t)
        else:
            scheduler.gain_schedule(3, t)
            print(f"End of Ramp Time: {scheduler.t_ramp_end}")

        scheduler.print_gains()
        t += 0.01

    
    assert scheduler.gains.KP == 1.5
    assert scheduler.gains.KI == 7.6
    assert scheduler.gains.KD == 4.9

    print("Gain transition test passed!")


if __name__ == "__main__":
    # manual_sim()
    # simple_transition()
    # change_gains_in_middle_01()
    # change_gains_in_middle_02()
    change_gains_in_middle_03()
