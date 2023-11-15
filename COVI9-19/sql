select c.class_name, sum(v.violations) from violations v, classroom c
where v.class_name = c.class_name group by v.class_name;

 select  sum(v.violations), strftime('%m-%d-%Y', start_of_detection) from violations v, classroom c where v.class_name = c.class_name group by strftime('%m-%d
-%Y', start_of_detection);

 select strftime('%m', start_of_detection), sum(v.violations),sum(v.social_violations),sum(v.total_violations), c.class_name from violations v, classroom c where v.class_name='room b' and v.class_name = c.class_name group by strftime('%m', start_of_detection);

 select strftime('%y', start_of_detection), sum(v.violations), sum(v.social_violations),sum(v.total_violations), c.class_name from violations v, classroom c where v.class_name='room b' and v.class_name = c.class_name group by strftime('%y', start_of_detection);

 select strftime('%d-%m', start_of_detection), sum(v.violations),sum(v.social_violations),sum(v.total_violations), c.class_name from violations v, classroom c where v.class_name='room b' and v.class_name = c.class_name group by strftime('%d-%m', start_of_detection) order by strftime('%m', start_of_detection) ASC;

 select c.class_name,sum(v.total_violations) from violations v, classroom c where v.class_name = c.class_name group by c.class_name order by sum(v.total_violations);
