
def classroom(opts, overclock_tick_ms):
  substitutions = {"RENDER_WIDTH": opts.width,
                   "RENDER_HEIGHT": opts.height,
                   "EPISODE_TIME_MS": (opts.episode_time_sec * 1000),
                   "MS_PER_TICK": overclock_tick_ms,
                   "OFFSCREEN_RENDERING": not opts.onscreen_rendering}

  if opts.mission == 1:
    substitutions.update({"WIDTH_LENGTH": 7,
                          "PATH_LENGTH": 0,
                          "DIVISIONS": 0,
                          "GAP_RATIO": 0,
                          "BRIDGE_RATIO": 0})
  elif opts.mission == 2:
    # block is one room away
    substitutions.update({"WIDTH_LENGTH": 15,
                          "PATH_LENGTH": 1,
                          "DIVISIONS": 1,
                          "GAP_RATIO": 1000,
                          "BRIDGE_RATIO": 0})
  elif opts.mission == 3:
    # block is three rooms away and there are occasional lava bridges
    substitutions.update({"WIDTH_LENGTH": 23,
                          "PATH_LENGTH": 3,
                          "DIVISIONS": 3,
                          "GAP_RATIO": 1000,
                          "BRIDGE_RATIO": 100})
  else:
    raise Exception("unknown mission %d" % opts.mission)

  spec = open("classroom_template.xml").read()
  for k, v in substitutions.iteritems():
    k = "__" + k + "__"
    assert k in spec
    spec = spec.replace(k, str(v).lower())
  return spec
