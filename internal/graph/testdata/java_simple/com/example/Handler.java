package com.example;

public class Handler {
    private String name;

    public Handler(String name) {
        this.name = name;
    }

    public String greet() {
        return this.format(this.name);
    }

    public String format(String x) {
        return "Hello, " + x;
    }
}
