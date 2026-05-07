library(shiny)

ui <- fluidPage(
  "Hello, world!"
)
server <- function(input, output, session) {
  
}

ui <- fluidPage(
  selectInput("dataset", 
                label = "Dataset", 
                choices = ls("package:datasets")),
                verbatimTextOutput("summary"),
                tableOutput("table")
)


#----------------  Method 1: Directly Using Bodyfat Column Names

ui <- fluidPage(
  selectInput("variable", 
              label = "Select Bodyfat Variable", 
              choices = names(Bodyfat)),
  verbatimTextOutput("summary"),
  tableOutput("table")
)

server <- function(input, output) {
  output$summary <- renderPrint({
    # Summary statistics for selected variable
    summary(Bodyfat[[input$variable]])
  })
  
  output$table <- renderTable({
    # Display first 10 rows of selected variable
    head(Bodyfat[input$variable], 10)
  })
}





#------------- Method 2: More Detailed with Data Type Filtering


ui <- fluidPage(
  selectInput("variable", 
              label = "Select Bodyfat Variable", 
              choices = names(Bodyfat)[sapply(Bodyfat, is.numeric)]), # Only numeric variables
  plotOutput("histogram"),
  verbatimTextOutput("summary"),
  tableOutput("table")
)

server <- function(input, output) {
  output$histogram <- renderPlot({
    req(input$variable)
    hist(Bodyfat[[input$variable]], 
         main = paste("Distribution of", input$variable),
         xlab = input$variable,
         col = "lightblue")
  })
  
  output$summary <- renderPrint({
    req(input$variable)
    summary(Bodyfat[[input$variable]])
  })
  
  output$table <- renderTable({
    req(input$variable)
    head(Bodyfat[c("Age", input$variable)], 10) # Show Age alongside selected variable
  })
}



#------------------  Method 3: Multiple Selections for Analysis

ui <- fluidPage(
  selectInput("x_var", "X Variable:", choices = names(Bodyfat)[sapply(Bodyfat, is.numeric)]),
  selectInput("y_var", "Y Variable:", choices = names(Bodyfat)[sapply(Bodyfat, is.numeric)]),
  plotOutput("scatterplot"),
  verbatimTextOutput("correlation"),
  tableOutput("regression")
)

server <- function(input, output) {
  output$scatterplot <- renderPlot({
    req(input$x_var, input$y_var)
    plot(Bodyfat[[input$x_var]], Bodyfat[[input$y_var]],
         xlab = input$x_var, ylab = input$y_var,
         main = paste(input$y_var, "vs", input$x_var),
         pch = 16, col = "blue")
    # Add regression line
    if(input$x_var != input$y_var) {
      abline(lm(Bodyfat[[input$y_var]] ~ Bodyfat[[input$x_var]]), col = "red")
    }
  })
  
  output$correlation <- renderPrint({
    req(input$x_var, input$y_var)
    if(input$x_var != input$y_var) {
      cor_val <- cor(Bodyfat[[input$x_var]], Bodyfat[[input$y_var]])
      cat("Correlation coefficient:", round(cor_val, 3))
    }
  })
  
  output$regression <- renderTable({
    req(input$x_var, input$y_var)
    if(input$x_var != input$y_var) {
      model <- lm(Bodyfat[[input$y_var]] ~ Bodyfat[[input$x_var]])
      summary(model)$coefficients
    }
  })
}



shinyApp(ui, server)
