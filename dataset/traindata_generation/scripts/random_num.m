function random_numbers_sorted = generate_sorted_random_numbers(num_values, min_value, max_value)
    % 定义生成随机数的参数
    min_difference = 2;

    % 初始化随机数数组
    random_numbers = zeros(1, num_values);

    % 生成第一个随机数
    random_numbers(1) = (max_value - min_value) * rand() + min_value;

    for i = 2:num_values
        while true
            % 生成新的候选随机数
            candidate = (max_value - min_value) * rand() + min_value;

            % 检查是否与之前的所有随机数相差至少 min_difference
            if all(abs(candidate - random_numbers(1:i-1)) >= min_difference)
                random_numbers(i) = candidate;
                break;
            end
        end
    end

    % 对生成的随机数从小到大排序
    random_numbers_sorted = sort(random_numbers);
end
