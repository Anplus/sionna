        # Build the rays for shooting
        # Origins
        # [max_depth, num_targets, num_targets, num_paths, 3]
        hit_points = tf.tile(tf.expand_dims(hit_points, axis=1),
                              [1, num_targets, 1, 1, 1])
        # [max_depth * num_targets * num_sources * num_paths, 3]
        ray_origins = tf.reshape(hit_points, [-1, 3])