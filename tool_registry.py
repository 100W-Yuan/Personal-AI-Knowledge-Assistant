class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, name, function, definition):
        self.tools[name] = {
            "function": function,
            "definition": definition
        }

    def get_definitions(self):
        return [
            tool["definition"]
            for tool in self.tools.values()
        ]

    def get_openai_tools(self):
        """
        返回 OpenAI / DeepSeek Chat Completions API
        所需要的 tools 格式。
        """

        return [
            {
                "type": "function",
                "function": tool["definition"]
            }
            for tool in self.tools.values()
        ]

    def validate_arguments(self, name, arguments):
        """
        在真正执行工具之前，检查工具参数是否符合要求。
        """

        # 1. 检查工具是否存在
        if name not in self.tools:
            return {
                "valid": False,
                "error": f"未知工具：{name}"
            }

        # 2. arguments 必须是字典
        if not isinstance(arguments, dict):
            return {
                "valid": False,
                "error": "工具参数必须是字典"
            }

        # 3. 读取工具的参数定义
        definition = self.tools[name]["definition"]

        parameters = definition.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        # 4. 检查必填参数
        for parameter_name in required:
            if parameter_name not in arguments:
                return {
                    "valid": False,
                    "error": f"缺少必填参数：{parameter_name}"
                }

        # 5. 检查参数类型
        for parameter_name, value in arguments.items():

            # 参数不在定义里
            if parameter_name not in properties:
                return {
                    "valid": False,
                    "error": f"未知参数：{parameter_name}"
                }

            parameter_definition = properties[parameter_name]
            expected_type = parameter_definition.get("type")

            # 字符串
            if expected_type == "string":
                if not isinstance(value, str):
                    return {
                        "valid": False,
                        "error": (
                            f"参数 {parameter_name} 类型错误："
                            f"需要字符串，实际是 {type(value).__name__}"
                        )
                    }

            # 整数
            elif expected_type == "integer":
                if not isinstance(value, int) or isinstance(value, bool):
                    return {
                        "valid": False,
                        "error": (
                            f"参数 {parameter_name} 类型错误："
                            f"需要整数，实际是 {type(value).__name__}"
                        )
                    }

            # 数字
            elif expected_type == "number":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return {
                        "valid": False,
                        "error": (
                            f"参数 {parameter_name} 类型错误："
                            f"需要数字，实际是 {type(value).__name__}"
                        )
                    }

            # 布尔值
            elif expected_type == "boolean":
                if not isinstance(value, bool):
                    return {
                        "valid": False,
                        "error": (
                            f"参数 {parameter_name} 类型错误："
                            f"需要布尔值，实际是 {type(value).__name__}"
                        )
                    }

        # 所有检查通过
        return {
            "valid": True,
            "error": None
        }

    def execute(self, name, arguments):

        # 第一步：参数校验
        validation = self.validate_arguments(name, arguments)

        if not validation["valid"]:
            return {
                "success": False,
                "error": f"工具参数校验失败：{validation['error']}"
            }

        # 第二步：取得工具函数
        function = self.tools[name]["function"]

        # 第三步：真正执行工具
        try:
            result = function(**arguments)

            return {
                "success": True,
                "result": result
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc)
            }