module subroutines
    implicit none
 contains 

    REAL FUNCTION sigmoid(input) RESULT(result)
    REAL, INTENT(IN) :: input
    result = (1.0 / (1.0 + exp(0.0 - input)))
    if (ISNAN(result)) then
        result = 5.0
        print *,"NOT A NUMBER"
    end if
    END FUNCTION sigmoid

    REAL FUNCTION rootMeanSquaredError(target_arr, output_arr) RESULT(result)
    REAL, INTENT(IN), DIMENSION(:) :: target_arr, output_arr

    REAL square, difference
    INTEGER i
    
    square = 0.0
    do i = 1,SIZE(target_arr)
        difference = target_arr(i) - output_arr(i)
        square = square + difference**2.0
    end do
    result = (square / i)**0.5
    !RETURN
    END FUNCTION rootMeanSquaredError

    SUBROUTINE feedForward(output_mat, input_arr, num_layers,&
        layer_size, weight_mat)
    INTEGER, INTENT(IN) :: num_layers
    INTEGER, INTENT(IN) :: layer_size(num_layers)
    REAL, INTENT(IN), DIMENSION(:) :: input_arr
    REAL, INTENT(IN), DIMENSION(:,:,:) :: weight_mat
    REAL, INTENT(INOUT), DIMENSION(:,:) :: output_mat

    REAL sum
    INTEGER i, j, k

    do i = 1, layer_size(1) !assign content to input layer
        output_mat(1,i) = input_arr(i)
    end do

    do i = 2, num_layers
        do j = 1, layer_size(i)
            sum = 0.0
            do k = 1, layer_size(i-1)
                sum = sum + (output_mat(j,(i-1)) * weight_mat(i,j,k))
            end do
            sum = sum + weight_mat(i,j,(layer_size(i-1)))
            sum = sigmoid(sum)

            output_mat(i,j) = sum
        end do
    end do
    END SUBROUTINE feedForward

    SUBROUTINE findDeltas(delta_err_mat, target_arr, weight_mat,&
        output_mat, num_layers, layer_size, target_num)
    INTEGER, INTENT(IN) :: num_layers
    INTEGER, INTENT(IN) :: layer_size(num_layers)
    REAL, INTENT(IN), DIMENSION(:) :: target_arr
    REAL, INTENT(IN), DIMENSION(:,:,:) :: weight_mat
    REAL, INTENT(IN), DIMENSION(:,:) :: output_mat
    REAL, INTENT(INOUT), DIMENSION(:,:) :: delta_err_mat

    INTEGER i,j,k, target_num
    REAL derr, omat, targ, sum

    !find output deltas
    do i = 1, layer_size(num_layers)
        omat = 0.0
        targ = 0.0
        derr = 0.0
        omat = output_mat((num_layers),i)
        targ = target_arr(target_num)

        derr = (targ - omat) * (sigmoid(omat) * (1.0 - sigmoid(omat)))

        delta_err_mat((num_layers),i) = derr
    end do

    !find hidden deltas
    do i = (num_layers-1),1,-1
        do j = 1,layer_size(i)
            sum = 0.0
            derr = 0.0
            do k = 1, (layer_size(i+1))
                sum = sum + delta_err_mat((i+1),k)*weight_mat((i+1),k,j)
            end do
            derr = sum * (output_mat(i,j) * (1 - output_mat(i,j)))
            delta_err_mat(i,j) = derr
        end do
    end do
    END SUBROUTINE findDeltas

    FUNCTION applyMomentum(weight_mat, prev_weight_mat, num_layers,&
     layer_size, alpha) RESULT(weight_matrix)
    INTEGER, INTENT(IN) :: num_layers
    INTEGER, INTENT(IN) :: layer_size(num_layers)
    REAL, INTENT(IN) :: alpha
    REAL, INTENT(IN), DIMENSION(:,:,:) :: prev_weight_mat
    REAL, INTENT(IN), DIMENSION(:,:,:) :: weight_mat

    INTEGER i, j, k

    REAL, DIMENSION(SIZE(weight_mat,1),&
        SIZE(weight_mat,2),&
        SIZE(weight_mat,3)) :: weight_matrix

    do i = 2, num_layers
        do j = 1, layer_size(i)
            do k = 1, layer_size(i-1)
                weight_matrix(i,j,k) = weight_mat(i,j,k) - alpha * &
                prev_weight_mat(i,j,k)
            end do
            weight_matrix(i,j,(layer_size(i-1))) = &
            weight_mat(i,j,(layer_size(i-1)))-alpha*&
            prev_weight_mat(i,j,(layer_size(i-1)))
        end do
    end do
    END FUNCTION applyMomentum

    SUBROUTINE adjustWeights(weight_mat, prev_weight_mat, num_layers,&
     layer_size, delta_err_mat, output_mat, beta)
    INTEGER, INTENT(IN) :: num_layers
    INTEGER, INTENT(IN) :: layer_size(num_layers)
    REAL, INTENT(IN), DIMENSION(:,:) :: output_mat, delta_err_mat
    REAL, INTENT(IN) :: beta
    REAL, INTENT(INOUT), DIMENSION(:,:,:) :: weight_mat, prev_weight_mat

    INTEGER i, j, k
    REAL weight_old

    do i = 2, num_layers
        do j = 1, layer_size(i)
            !print *,"derr: ",delta_err_mat(i,j)
            weight_old = 0.0
            do k = 1, layer_size(i-1)
                !print *,"derr ",delta_err_mat(i,j)," out ",output_mat((i-1),k)
                prev_weight_mat(i,j,k) = beta * delta_err_mat(i,j) * &
                output_mat((i-1),k)
                weight_mat(i,j,k) = weight_mat(i,j,k) + prev_weight_mat(i,j,k)
            end do
            weight_old = beta * delta_err_mat(i,j)
            prev_weight_mat(i,j,(layer_size(i-1))) = weight_old

            weight_mat(i,j,(layer_size(i-1))) = &
            weight_mat(i,j,(layer_size(i-1))) + weight_old


            !print *,"i: ",i,"j: ",j,"old weight: ",weight_old," new weight: ",weight_new
        end do
    end do

    END SUBROUTINE adjustWeights

    SUBROUTINE backPropagate(input_arr,target_arr,output_mat,num_layers,&
        layer_size,weight_mat,delta_err_mat,prev_weight_mat,a,b, target_num)
    INTEGER, INTENT(IN) :: num_layers, target_num
    INTEGER, INTENT(IN) :: layer_size(num_layers)
    REAL, INTENT(IN), DIMENSION(:) :: input_arr, target_arr
    REAL, INTENT(IN) :: a, b
    REAL, INTENT(INOUT), DIMENSION(:,:) :: output_mat, delta_err_mat
    REAL, INTENT(INOUT), DIMENSION(:,:,:) :: weight_mat, prev_weight_mat

    INTEGER i,j,k

    CALL feedForward(output_mat, input_arr, num_layers, layer_size, weight_mat)

    CALL findDeltas(delta_err_mat, target_arr, weight_mat,&
        output_mat, num_layers, layer_size, target_num)

    weight_mat = applyMomentum(weight_mat, prev_weight_mat,&
        num_layers, layer_size, a)

    CALL adjustWeights(weight_mat, prev_weight_mat, num_layers,&
        layer_size, delta_err_mat, output_mat, b)

    END SUBROUTINE backPropagate
end module

PROGRAM main

use subroutines

!REMINDER: fortran arrays/matrices are in column-major order

!REMINDER: declarations MUST come before assignments

INTEGER layer_size(5), num_layers, num_iterations
REAL b, a, threshold, rmse, outputs(8), old_rmse
REAL, DIMENSION(:,:), allocatable :: output_mat, out_matrix, delta_err_mat
REAL, DIMENSION(:,:,:), allocatable :: weight_mat, prev_weight_mat
INTEGER i, j, k, rmse_count
REAL rand_val, out_val

REAL, DIMENSION(8, 4) :: data = reshape((/ 0, 0, 0, 0, &
                                                    0, 0, 1, 1, &
                                                    0, 1, 0, 1, &
                                                    0, 1, 1, 0, &
                                                    1, 0, 0, 1, &
                                                    1, 0, 1, 0, &
                                                    1, 1, 0, 0, &
                                                    1, 1, 1, 1/), &
                                                    shape(data), order=(/2,1/))

REAL, DIMENSION(8, 3) :: training_data = reshape((/ 0, 0, 0, &
                                                        0, 0, 1, &
                                                        0, 1, 0, &
                                                        0, 1, 1, &
                                                        1, 0, 0, &
                                                        1, 0, 1, &
                                                        1, 1, 0, &
                                                        1, 1, 1/), &
                                                        shape(training_data),&
                                                        order=(/2,1/))

layer_size = (/3, 3, 3, 3, 1/)
outputs = (/8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0/)
num_layers = 5
num_iterations = 999999    !this number modulo 8 should return 7
                                     !this is to ensure all the data is processed
beta = 0.01
alpha = 0.0
threshold = 0.0001
rmse = 1.0

!TODO: figure out how to allocate this more neatly
allocate(output_mat(num_layers,layer_size(1)))
allocate(out_matrix(num_layers,layer_size(1)))
allocate(delta_err_mat(num_layers,layer_size(1)))

output_mat = RESHAPE(output_mat,SHAPE(output_mat),order=(/2,1/))
delta_err_mat = RESHAPE(delta_err_mat,SHAPE(delta_err_mat),order=(/2,1/))

do i = 1, num_layers
    do j = 1, layer_size(i)
        output_mat(i,j) = 0.0
        out_matrix(i,j) = 0.0
        delta_err_mat(i,j) = 0.0
    end do
end do

allocate(weight_mat((layer_size(1)+1),layer_size(1),num_layers))
allocate(prev_weight_mat((layer_size(1)+1),layer_size(1),num_layers))

do i = 2, num_layers
    do j = 1, layer_size(i)
        do k = 1, (layer_size(i-1)+1)
            !print *,"i ",i," j ",j," k ",k
            rand_val = RAND()
            !print *,rand_val
            if(rand_val <= 0.001) then
                rand_val = 0.001
            end if
            weight_mat(i,j,k) = rand_val - 1.0
            prev_weight_mat(i,j,k) = 0.0
        end do
    end do
end do

weight_mat = RESHAPE(weight_mat,SHAPE(weight_mat),order=(/3,2,1/))
prev_weight_mat=RESHAPE(prev_weight_mat,SHAPE(prev_weight_mat),order=(/3,2,1/))

rmse_count = 0
do i = 1,8
    do j = 0,num_iterations
        out_val = 0.0
        CALL backPropagate(data(i,:),data(:,4),output_mat,&
            num_layers,layer_size,weight_mat,delta_err_mat,prev_weight_mat,a,b,i)

        out_val = output_mat(num_layers,1)
        outputs(i) = out_val
        
        !TODO: disable this if the rmse is below 0.01
        rmse = rootMeanSquaredError(data(:,4), outputs)
        if(old_rmse == rmse) then
            rmse_count = rmse_count + 1
            if(rmse_count >= 100) then
                CALL random_number(weight_mat)
                rmse_count = 0
            end if
        end if
        old_rmse = rmse
        if(rmse < threshold) then
            print *,"DONE at iteration ",j," of i: ",i
            exit
        end if
        !if(rmse > rmse+1) exit
        !if(ISNAN(rmse)) exit
        !print *,"rmse: ",rmse
    end do
    if(rmse < threshold) then
        print *,"DONE at iteration ",j," of i: ",i
        exit
    end if
end do
print *," "
print *,data(1,1)," ",data(1,2)," ",data(1,3)," ",data(1,4)
print *,data(2,1)," ",data(2,2)," ",data(2,3)," ",data(2,4)
print *,data(3,1)," ",data(3,2)," ",data(3,3)," ",data(3,4)
print *,data(4,1)," ",data(4,2)," ",data(4,3)," ",data(4,4)
print *,data(5,1)," ",data(5,2)," ",data(5,3)," ",data(5,4)
print *,data(6,1)," ",data(6,2)," ",data(6,3)," ",data(6,4)
print *,data(7,1)," ",data(7,2)," ",data(7,3)," ",data(7,4)
print *,data(8,1)," ",data(8,2)," ",data(8,3)," ",data(8,4)
print *," "

!test the trained network
out_matrix = output_mat
do i=1,8
    CALL feedForward(out_matrix,training_data(i,:),num_layers,layer_size,weight_mat)
    outputs(i) = out_matrix(num_layers,1)
end do

print *,"results: "
do i=1,8
    print *, training_data(i,1)," ",training_data(i,2)," ",training_data(i,3),&
    " ",outputs(i)
end do

deallocate(output_mat, out_matrix,delta_err_mat)
deallocate(prev_weight_mat, weight_mat)
END PROGRAM main
