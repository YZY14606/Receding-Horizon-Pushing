import numpy as np
import object_planner_py as opp
import open3d as o3d
import typing




class Path_planner():
    """This planner is used to plan a path of the pushed object.
    The pipline of the ptah planner is: 
    1. __init__()
    2. set_up_planner()
    3. plan_path()
    """
    def __init__(
        self,
        planner_params: typing.Dict,
        state_cost_settings=None,
    ):
        max_batches = planner_params.get("max_batches", 20)
        samples_per_batch = planner_params.get("samples_per_batch", 200)
        collision_check_resolution = planner_params.get("collision_check_resolution", 0.01)
        shortcut_enabled = planner_params.get("simplify_path", True)
        shortcut_relative = planner_params.get(
            "simplify_max_relative_cost_increase",
            0.0,
        )
        shortcut_absolute = planner_params.get(
            "simplify_absolute_cost_tolerance",
            1e-9,
        )
        self.point_inflation = planner_params.get("point_inflation", 0.0)
        self.active_point_inflation = None

        self.bit_star_params = self.set_up_planner_params(
            max_batches,
            samples_per_batch,
            collision_check_resolution,
            shortcut_enabled,
            shortcut_relative,
            shortcut_absolute,
        )
        self.state_cost_settings = (
            state_cost_settings
            if state_cost_settings is not None
            else opp.StateCostSettings()
        )
        self.failed_configs = []
        self.failed_transitions = []
        self.active_failed_configs = ()
        self.active_failed_transitions = ()
        self.active_state_cost_settings = None
        self.path_planner = None
        self.solution_cost = np.inf
        self.raw_solution_cost = np.inf
        self.simplification_stats = None

        self.path = None
        self.obstacle_points = None
        self.object_points = None


    
    def set_up_planner_params(
        self,
        max_batches,
        samples_per_batch,
        collision_check_resolution,
        shortcut_enabled,
        shortcut_relative,
        shortcut_absolute,
    ):

        plan_params = opp.BITStarParams()
        plan_params.max_batches = max_batches
        plan_params.samples_per_batch = samples_per_batch
        plan_params.collision_check_resolution = collision_check_resolution

        shortcut = opp.CostAwareShortcutSettings()
        shortcut.enabled = shortcut_enabled
        shortcut.max_relative_cost_increase = shortcut_relative
        shortcut.absolute_cost_tolerance = shortcut_absolute
        plan_params.shortcut_simplification = shortcut

        return plan_params

    def set_failure_context(self, failed_poses, failed_transitions):
        """Replace the pose and directed-transition failure context."""
        failed_snapshot = self._snapshot_failed_configs(failed_poses)
        transition_snapshot = self._snapshot_failed_transitions(
            failed_transitions
        )
        planner_failed_configs = [
            opp.Config(x, y, theta)
            for x, y, theta in failed_snapshot
        ]
        planner_failed_transitions = [
            opp.FailedTransition(
                start=opp.Config(*start),
                goal=opp.Config(*goal),
            )
            for start, goal in transition_snapshot
        ]

        if self.path_planner is not None:
            self.path_planner.set_failure_context(
                failed_poses=planner_failed_configs,
                failed_transitions=planner_failed_transitions,
            )

        self.failed_configs = planner_failed_configs
        self.failed_transitions = planner_failed_transitions
        if self.path_planner is not None:
            self.active_failed_configs = failed_snapshot
            self.active_failed_transitions = transition_snapshot

        self.path = None
        self.solution_cost = np.inf
        self.raw_solution_cost = np.inf
        self.simplification_stats = None

    def set_cost_context(self, failed_configs):
        """Compatibility wrapper for pose-only callers."""
        self.set_failure_context(failed_configs, [])

    @staticmethod
    def _validate_point_cloud(points, name, allow_empty):
        try:
            point_cloud = np.array(
                points,
                dtype=np.float64,
                order="C",
                copy=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be convertible to float64") from exc

        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"{name} must have shape (N, 3)")
        if not allow_empty and len(point_cloud) == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.all(np.isfinite(point_cloud)):
            raise ValueError(f"{name} must contain only finite values")
        point_cloud.setflags(write=True)
        return point_cloud

    @classmethod
    def _prepare_obstacle_points(cls, obstacle_source):
        if isinstance(obstacle_source, np.ndarray):
            return cls._validate_point_cloud(
                obstacle_source,
                "all_obstacle_points",
                allow_empty=True,
            )

        if not isinstance(obstacle_source, (list, tuple)):
            raise ValueError(
                "all_obstacle_points must be an (N, 3) array or a list "
                "of point clouds"
            )

        obstacle_clouds = []
        for index, point_cloud in enumerate(obstacle_source):
            validated_cloud = cls._validate_point_cloud(
                point_cloud,
                f"all_obstacle_points[{index}]",
                allow_empty=True,
            )
            if len(validated_cloud) > 0:
                obstacle_clouds.append(validated_cloud)

        if not obstacle_clouds:
            obstacle_points = np.empty((0, 3), dtype=np.float64)
            obstacle_points.setflags(write=False)
            return obstacle_points
        obstacle_points = np.array(
            np.vstack(obstacle_clouds),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        obstacle_points.setflags(write=False)
        return obstacle_points

    @staticmethod
    def _validate_bounds(bounds, name):
        try:
            bounds_array = np.asarray(bounds, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain two finite numbers") from exc

        if bounds_array.shape != (2,) or not np.all(np.isfinite(bounds_array)):
            raise ValueError(f"{name} must contain two finite numbers")
        if bounds_array[0] >= bounds_array[1]:
            raise ValueError(f"{name} lower bound must be smaller than upper bound")
        return (float(bounds_array[0]), float(bounds_array[1]))

    @staticmethod
    def _snapshot_failed_configs(failed_configs):
        snapshot = []
        for index, config in enumerate(failed_configs):
            try:
                values = (
                    float(config.x),
                    float(config.y),
                    float(config.theta),
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"failed_configs[{index}] must expose finite x, y, theta"
                ) from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"failed_configs[{index}] must expose finite x, y, theta"
                )
            snapshot.append(values)
        return tuple(snapshot)

    @staticmethod
    def _snapshot_failed_transitions(failed_transitions):
        snapshot = []
        for index, transition in enumerate(failed_transitions):
            try:
                endpoints = (transition.start, transition.goal)
                values = tuple(
                    (
                        float(endpoint.x),
                        float(endpoint.y),
                        float(endpoint.theta),
                    )
                    for endpoint in endpoints
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"failed_transitions[{index}] must expose finite "
                    "start and goal Config values"
                ) from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"failed_transitions[{index}] must expose finite "
                    "start and goal Config values"
                )
            snapshot.append(values)
        return tuple(snapshot)

    @staticmethod
    def _snapshot_state_cost_settings(settings):
        setting_names = (
            "avoid_radius_xy",
            "xy_sigma",
            "theta_sigma",
            "state_cost_peak",
            "max_state_cost",
            "transition_radius_xy",
            "transition_xy_sigma",
            "direction_sigma",
            "transition_cost_peak",
            "max_transition_cost",
            "max_total_cost",
            "min_transition_xy_length",
        )
        snapshot = {}
        for name in setting_names:
            try:
                value = float(getattr(settings, name))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"state_cost_settings.{name} must be finite"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"state_cost_settings.{name} must be finite"
                )
            snapshot[name] = value
        return snapshot

    @staticmethod
    def _state_cost_settings_from_snapshot(snapshot):
        settings = opp.StateCostSettings()
        for name, value in snapshot.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _create_line_set(points, lines, color):
        points_array = np.array(
            points,
            dtype=np.float64,
            order="C",
            copy=True,
        ).reshape(-1, 3)
        lines_array = np.array(
            lines,
            dtype=np.int32,
            order="C",
            copy=True,
        ).reshape(-1, 2)
        color_array = np.asarray(color, dtype=np.float64)

        if not np.all(np.isfinite(points_array)):
            raise ValueError("line-set points must contain only finite values")
        if (
            len(lines_array) > 0
            and (
                np.any(lines_array < 0)
                or np.any(lines_array >= len(points_array))
            )
        ):
            raise ValueError("line-set indices must reference existing points")
        if color_array.shape != (3,) or not np.all(np.isfinite(color_array)):
            raise ValueError("line-set color must contain three finite values")

        points_array.setflags(write=True)
        lines_array.setflags(write=True)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_array),
            lines=o3d.utility.Vector2iVector(lines_array),
        )
        if len(lines_array) > 0:
            colors = np.array(
                np.tile(color_array, (len(lines_array), 1)),
                dtype=np.float64,
                order="C",
                copy=True,
            )
            colors.setflags(write=True)
            line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    @classmethod
    def _create_failed_config_geometries(
        cls,
        failed_configs,
        settings,
        overlay_z=0.045,
        ring_count=128,
        spoke_count=16,
    ):
        if (
            not isinstance(ring_count, (int, np.integer))
            or isinstance(ring_count, (bool, np.bool_))
            or int(ring_count) < 3
        ):
            raise ValueError("ring_count must be an integer of at least 3")
        if (
            not isinstance(spoke_count, (int, np.integer))
            or isinstance(spoke_count, (bool, np.bool_))
            or int(spoke_count) < 3
        ):
            raise ValueError("spoke_count must be an integer of at least 3")

        overlay_z = float(overlay_z)
        if not np.isfinite(overlay_z):
            raise ValueError("overlay_z must be finite")
        try:
            radius = float(settings["avoid_radius_xy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "settings must contain a positive avoid_radius_xy"
            ) from exc
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("settings avoid_radius_xy must be positive")

        ring_count = int(ring_count)
        spoke_count = int(spoke_count)
        ring_angles = np.linspace(
            0.0,
            2.0 * np.pi,
            ring_count,
            endpoint=False,
        )
        spoke_angles = np.linspace(
            0.0,
            2.0 * np.pi,
            spoke_count,
            endpoint=False,
        )
        geometries = []

        for failed_index, failed_config in enumerate(failed_configs):
            failed_array = np.asarray(failed_config, dtype=np.float64)
            if (
                failed_array.shape != (3,)
                or not np.all(np.isfinite(failed_array))
            ):
                raise ValueError(
                    f"failed_configs[{failed_index}] must contain finite "
                    "x, y, theta"
                )
            failed_x, failed_y, _failed_theta = failed_array

            ring_points = np.column_stack(
                (
                    failed_x + radius * np.cos(ring_angles),
                    failed_y + radius * np.sin(ring_angles),
                    np.full(ring_count, overlay_z),
                )
            )
            ring_lines = [
                [index, (index + 1) % ring_count]
                for index in range(ring_count)
            ]
            geometries.append(
                cls._create_line_set(
                    ring_points,
                    ring_lines,
                    [1.0, 0.15, 0.0],
                )
            )

            spoke_endpoints = np.column_stack(
                (
                    failed_x + radius * np.cos(spoke_angles),
                    failed_y + radius * np.sin(spoke_angles),
                    np.full(spoke_count, overlay_z),
                )
            )
            spoke_points = np.vstack(
                (
                    [failed_x, failed_y, overlay_z],
                    spoke_endpoints,
                )
            )
            spoke_lines = [
                [0, index]
                for index in range(1, spoke_count + 1)
            ]
            geometries.append(
                cls._create_line_set(
                    spoke_points,
                    spoke_lines,
                    [1.0, 0.65, 0.05],
                )
            )

            failed_center = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.025
            )
            failed_center.translate(
                [failed_x, failed_y, overlay_z + 0.01]
            )
            failed_center.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(failed_center)

        return geometries

    @classmethod
    def _create_failed_transition_geometries(
        cls,
        failed_transitions,
        overlay_z=0.065,
    ):
        geometries = []
        for index, transition in enumerate(failed_transitions):
            transition_array = np.asarray(transition, dtype=np.float64)
            if (
                transition_array.shape != (2, 3)
                or not np.all(np.isfinite(transition_array))
            ):
                raise ValueError(
                    f"failed_transitions[{index}] must contain finite "
                    "start and goal Config values"
                )

            start_xy = transition_array[0, :2]
            goal_xy = transition_array[1, :2]
            delta = goal_xy - start_xy
            length = np.linalg.norm(delta)
            if length <= 1e-9:
                continue

            direction = delta / length
            normal = np.array([-direction[1], direction[0]])
            arrow_head = min(0.035, 0.4 * length)
            arrow_points_xy = np.vstack(
                (
                    start_xy,
                    goal_xy,
                    goal_xy - arrow_head * direction
                    + 0.5 * arrow_head * normal,
                    goal_xy - arrow_head * direction
                    - 0.5 * arrow_head * normal,
                )
            )
            arrow_points = np.column_stack(
                (arrow_points_xy, np.full(4, overlay_z))
            )
            geometries.append(
                cls._create_line_set(
                    arrow_points,
                    [[0, 1], [1, 2], [1, 3]],
                    [0.65, 0.0, 0.0],
                )
            )
        return geometries
    
    def set_up_planner(self,scene_dic: typing.Dict):
        """scene_dic includes:
        object_points: np.ndarray, # shape (N, 3)
        all_obstacle_points: np.ndarray or typing.List[np.ndarray]
        map_x_bounds: typing.Tuple[float, float] = (-1.5, 1.5),  
        map_y_bounds: typing.Tuple[float, float] = (-1.5, 1.5),  
        map_theta_bound: typing.Tuple[float, float] = (-np.pi, np.pi)
        point_inflation: non-negative finite object-obstacle clearance
        """
        object_points = self._validate_point_cloud(
            scene_dic["object_points"],
            "object_points",
            allow_empty=False,
        )
        obstacle_points = self._prepare_obstacle_points(
            scene_dic["all_obstacle_points"]
        )
        map_x_bounds = self._validate_bounds(
            scene_dic["map_x_bounds"],
            "map_x_bounds",
        )
        map_y_bounds = self._validate_bounds(
            scene_dic["map_y_bounds"],
            "map_y_bounds",
        )
        map_theta_bound = self._validate_bounds(
            scene_dic["map_theta_bound"],
            "map_theta_bound",
        )

        point_inflation = scene_dic.get(
            "point_inflation",
            self.point_inflation,
        )
        if isinstance(point_inflation, (bool, np.bool_)):
            raise ValueError("point_inflation must be a finite non-negative number")
        try:
            point_inflation = float(point_inflation)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "point_inflation must be a finite non-negative number"
            ) from exc
        if not np.isfinite(point_inflation) or point_inflation < 0:
            raise ValueError(
                "point_inflation must be a finite non-negative number"
            )

        # The C++ checker snapshots all inputs at construction time. Build it
        # first and only publish the new Python-side state after construction
        # succeeds, so a bad rebuild cannot corrupt a previously valid planner.
        failed_snapshot = self._snapshot_failed_configs(
            self.failed_configs
        )
        transition_snapshot = self._snapshot_failed_transitions(
            self.failed_transitions
        )
        settings_snapshot = self._snapshot_state_cost_settings(
            self.state_cost_settings
        )
        planner_failed_configs = [
            opp.Config(x, y, theta)
            for x, y, theta in failed_snapshot
        ]
        planner_failed_transitions = [
            opp.FailedTransition(
                start=opp.Config(*start),
                goal=opp.Config(*goal),
            )
            for start, goal in transition_snapshot
        ]
        planner_state_cost_settings = (
            self._state_cost_settings_from_snapshot(settings_snapshot)
        )
        new_planner = opp.Planner(
            object_points=object_points,
            obstacle_points=obstacle_points,
            x_bounds=map_x_bounds,
            y_bounds=map_y_bounds,
            theta_bounds=map_theta_bound,
            point_inflation=point_inflation,
            cost_context=planner_failed_configs,
            state_cost_settings=planner_state_cost_settings,
            failed_transitions=planner_failed_transitions,
        )

        canonical_scene = dict(scene_dic)
        canonical_scene["object_points"] = object_points
        canonical_scene["all_obstacle_points"] = obstacle_points
        canonical_scene["map_x_bounds"] = map_x_bounds
        canonical_scene["map_y_bounds"] = map_y_bounds
        canonical_scene["map_theta_bound"] = map_theta_bound
        canonical_scene["point_inflation"] = point_inflation

        self.path_planner = new_planner
        self.scene_dic = canonical_scene
        self.obstacle_points = obstacle_points
        self.object_points = object_points
        self.active_point_inflation = point_inflation
        self.active_failed_configs = failed_snapshot
        self.active_failed_transitions = transition_snapshot
        self.active_state_cost_settings = settings_snapshot
        self.path = None
        self.solution_cost = np.inf
        self.raw_solution_cost = np.inf
        self.simplification_stats = None
        
    def plan_path(self, start_config, goal_config):
        self.path = None
        self.solution_cost = np.inf
        self.raw_solution_cost = np.inf
        self.simplification_stats = None

        path = self.path_planner.plan_bit_star(
            start=start_config,
            goal=goal_config,
            plan_params=self.bit_star_params,
        )
        self.solution_cost = self.path_planner.bit_star_solution_cost()
        self.raw_solution_cost = (
            self.path_planner.bit_star_raw_solution_cost()
        )
        self.simplification_stats = (
            self.path_planner.bit_star_simplification_stats()
        )
        self.path = path
        return path
    

    def visualize_path(self,
        object_points,
        obstacle_points,
        path,
        start_config,
        goal_config,
        map_bounds_x,
        map_bounds_y,
        *,
        show_failed_configs=True,
    ):
        """
        Visualize the path and the failure context used by its planner.
        """
        active_failed_configs = (
            self.active_failed_configs
            if show_failed_configs
            else ()
        )
        active_failed_transitions = (
            self.active_failed_transitions
            if show_failed_configs
            else ()
        )
        has_path = path is not None and len(path) > 0
        if (
            not has_path
            and not active_failed_configs
            and not active_failed_transitions
        ):
            print("No path to visualize.")
            return

        if not has_path:
            print(
                "No path was found; displaying the active failure-cost "
                "context."
            )

        def create_object_at_config(p, c, clr):
            pcd = o3d.geometry.PointCloud()
            object_point_array = np.array(
                p,
                dtype=np.float64,
                order="C",
                copy=True,
            )
            object_point_array.setflags(write=True)
            pcd.points = o3d.utility.Vector3dVector(object_point_array)
            T = np.eye(4)
            T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, c.theta))
            T[0, 3] = c.x
            T[1, 3] = c.y
            pcd.transform(T)
            pcd.paint_uniform_color(clr)
            return pcd

        geometries = []
        obstacle_pcd = o3d.geometry.PointCloud()
        obstacle_point_array = np.array(
            obstacle_points,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        obstacle_point_array.setflags(write=True)
        obstacle_pcd.points = o3d.utility.Vector3dVector(
            obstacle_point_array
        )
        obstacle_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(obstacle_pcd)
        geometries.append(
            create_object_at_config(object_points, start_config, [0.0, 0.8, 0.2])
        )
        geometries.append(
            create_object_at_config(object_points, goal_config, [0.0, 0.2, 0.8])
        )
        if has_path:
            path_points_3d = [[c.x, c.y, 0.025] for c in path]
            if len(path_points_3d) > 1:
                geometries.append(
                    self._create_line_set(
                        path_points_3d,
                        [
                            [index, index + 1]
                            for index in range(len(path_points_3d) - 1)
                        ],
                        [0.0, 0.75, 1.0],
                    )
                )

            num_ghosts = 4
            if len(path) > num_ghosts + 1:
                path_without_ends = path[1:-1]
                step = len(path_without_ends) // num_ghosts
                indices_in_subpath = range(
                    0,
                    len(path_without_ends),
                    step if step > 0 else 1,
                )[:num_ghosts]
                for index in indices_in_subpath:
                    geometries.append(
                        create_object_at_config(
                            object_points,
                            path_without_ends[index],
                            [0.9, 0.7, 0.1],
                        )
                    )

            for point in path_points_3d:
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.015
                )
                sphere.translate(point)
                sphere.paint_uniform_color([0.6, 0.2, 0.8])
                geometries.append(sphere)

        if (
            active_failed_configs
            and self.active_state_cost_settings is not None
        ):
            geometries.extend(
                self._create_failed_config_geometries(
                    active_failed_configs,
                    self.active_state_cost_settings,
                )
            )
        if active_failed_transitions:
            geometries.extend(
                self._create_failed_transition_geometries(
                    active_failed_transitions
                )
            )

        map_w = map_bounds_x[1] - map_bounds_x[0]
        map_h = map_bounds_y[1] - map_bounds_y[0]
        table = o3d.geometry.TriangleMesh.create_box(width=map_w, height=map_h, depth=0.005)
        table.translate([map_bounds_x[0], map_bounds_y[0], -0.005])
        table.paint_uniform_color([0.8, 0.7, 0.6])
        geometries.append(table)
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        )
        print("This Step Displays Path Plan Visualization")
        if has_path:
            print("cyan path: current failure-aware BIT* path")
        if active_failed_configs:
            radius = self.active_state_cost_settings["avoid_radius_xy"]
            print(
                "orange ring/spokes: failed-config non-zero XY cost support"
            )
            print(
                "red center: direction-independent failed pose"
            )
            print(
                f"active failed context count: "
                f"{len(active_failed_configs)}"
            )
            for index, (failed_x, failed_y, failed_theta) in enumerate(
                active_failed_configs
            ):
                print(
                    f"failed[{index}]: x={failed_x:.4f}, "
                    f"y={failed_y:.4f}, theta={failed_theta:.4f} rad, "
                    f"radius={radius:.3f} m"
                )
        if active_failed_transitions:
            print("dark-red arrow: ordered failed transition")
            print(
                f"active failed transition count: "
                f"{len(active_failed_transitions)}"
            )
        o3d.visualization.draw_geometries(geometries)
